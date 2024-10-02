import argparse
import json
import pickle
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from Agent import Agent
from AuctionAllocation import * # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  #  LogisticTSAllocator, OracleAllocator
from sklearn.metrics.pairwise import cosine_similarity

from CombinatorialLinUCB import CombinatorialLinUCB
from Publisher import Publisher
from Publisher_Reward import PublisherReward


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Number of runs
    num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1

    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    embedding_size = config['embedding_size']
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config['agents']:
        if 'num_copies' in agent_config.keys():
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {num_agents + 1}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    # agents2items = {
    #     agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size))
    #     for agent_config in agent_configs
    # }

    agents2item_values = {
        agent_config['name']: rng.lognormal(-3, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }

    # Add intercepts to embeddings (Uniformly in [-4.5, -1.5], this gives nicer distributions for P(click))
    # for agent, items in agents2items.items():
    #     agents2items[agent] = np.hstack((items, - 3.0 - 1.0 * rng.random((items.shape[0], 1))))

    adv_embedding_path = config['adv_embedding_path']
    adv_embeddings = pickle.load(open(adv_embedding_path, 'rb'))
    agents2items = {
        agent_config['name']: adv_embeddings[agent_config['adv_name']]
        for agent_config in agent_configs
    }
    publisher_embeddings_path = config['publisher_embedding_path']
    publisher_embeddings = pickle.load(open(publisher_embeddings_path, 'rb'))
    # Knapsack parameters
    knapsack_params = config['knapsack_params']

    return (rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size,
            embedding_var, obs_embedding_size, adv_embeddings, publisher_embeddings, knapsack_params)


def instantiate_agents(rng, agent_configs, agents2item_values, agents2items):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              adv_name=agent_config['adv_name'],
              num_items=agent_config['num_items'],
              item_values=agents2item_values[agent_config['name']],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              memory=(0 if 'memory' not in agent_config.keys() else agent_config['memory']))
        for agent_config in agent_configs
    ]

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

    return agents


def instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size):
    return (Auction(rng,
                    eval(f"{config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    embedding_size,
                    embedding_var,
                    obs_embedding_size,
                    config['num_participants_per_round']),
            config['num_iter'], config['rounds_per_iter'], config['output_dir'])


def instantiate_publishers(publisher_embeddings):
    return [
        Publisher(
            name=publisher,
            embedding=embedding,
            num_auctions=500)
        for publisher, embedding in publisher_embeddings.items()
    ]


def add_noise(user_embedding, noise_strength=0.01):
    noise = np.random.normal(0, noise_strength, user_embedding.shape)
    return user_embedding + noise


def initialize_deal(publisher_list: List[Publisher], publisher_emb, round_per_iter, agents, adv_embeddings):
    chosen_publisher_emb = {publisher.name: publisher_emb[publisher.name] for publisher in publisher_list}
    # For each publisher, add noise to the embedding to simulate different users for rounds_per_iter times
    noisy_publisher_emb = {
        publisher.name: np.array([
            add_noise(chosen_publisher_emb[publisher.name])
            for _ in range(round_per_iter)
        ])
        for publisher in publisher_list
    }
    # Compute cosine similarity between each pair of noisy embeddings and all adv agents embeddings
    agents_publishers_similarity = {}
    for agent in agents:
        agents_publishers_similarity[agent.name] = {}
        agent_embedding = adv_embeddings[agent.adv_name]
        for publisher in publisher_list:
            noisy_emb = noisy_publisher_emb[publisher.name]
            agent_publisher_sim = cosine_similarity(noisy_emb, agent_embedding.reshape(1, -1))
            # If an element is negative, set it to 0
            agent_publisher_sim[agent_publisher_sim < 0] = 0
            agents_publishers_similarity[agent.name][publisher.name] = agent_publisher_sim
    return noisy_publisher_emb, agents_publishers_similarity


def simulation_run():
    # Choose 300 publishers at random
    rng.shuffle(publishers)
    init_publisher_list = publishers[:300]
    comb_linucb = CombinatorialLinUCB(alpha=1, d=embedding_size, publisher_list=init_publisher_list)
    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')
        if i > 1:
            # After the first two iterations, run Combinatorial LinUCB to change the publisher list
            start_comb_linucb = time.time()
            publisher_list = comb_linucb.round_iteration(init_publisher_list, iteration=i, soglia_spent=soglia_spent, soglia_clicks=soglia_clicks, soglia_cpc=soglia_cpc)
            print(f'Combinatorial LinUCB took {time.time() - start_comb_linucb} seconds')
        else:
            publisher_list = init_publisher_list
        start_comp_sim = time.time()
        # Compute all agents' similarity with all publishers
        user_contexts, agents_publishers_similarity = initialize_deal(
            publisher_list, publisher_embeddings, rounds_per_iter, agents, adv_embeddings)
        print(f'Computing similarity took {time.time() - start_comp_sim} seconds')
        start_sim_auctions = time.time()
        # For each publisher, simulate rounds_per_iter opportunities
        for publisher in publisher_list:
            for j in range(rounds_per_iter):
                current_user_context = user_contexts[publisher.name][j]
                auction.simulate_opportunity(publisher.name, current_user_context, agents_publishers_similarity, j)
        print(f'Simulating {len(publisher_list)*rounds_per_iter} auctions took {time.time() - start_sim_auctions} seconds')
        # Update agents bidder and allocator models
        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)
            # Our agent
            if agent.name.startswith('Nostro'):
                # Get stats for our agent
                agent_stats_pub = agent.iteration_stats_per_publisher()
                # Update Combinatorial LinUCB
                if i < 2:
                    # Initialize Combinatorial LinUCB parameters with the first two iterations
                    publishers_rewards = []
                    for publisher_data in agent_stats_pub:
                        curr_pub = Publisher(
                            name=publisher_data['publisher'],
                            embedding=publisher_embeddings[publisher_data['publisher']],
                            num_auctions=500
                        )
                        publishers_rewards.append(
                            PublisherReward(
                                publisher=curr_pub,
                                reward=publisher_data['num_clicks']
                            )
                        )
                    comb_linucb.initial_round(publishers_rewards, iteration=i)
                else:
                    # Update Combinatorial LinUCB after seeing the rewards
                    for publisher_data in agent_stats_pub:
                        comb_linucb.update(
                            publisher_name=publisher_data['publisher'],
                            publisher_embedding=publisher_embeddings[publisher_data['publisher']],
                            reward=publisher_data['num_clicks'],
                            iteration=i
                        )
                # Save stats for our agent
                agent_df = pd.DataFrame(agent_stats_pub)
                agent_df['Agent'] = agent.name
                agent_df['Iteration'] = i
                agent_df['Run'] = run
                filename = f'{output_dir}/agent_{agent.name}_stats.csv'
                if not os.path.exists(filename):
                    # If the file does not exist yet, write the header
                    agent_df.to_csv(filename, mode='a', header=True, index=False)
                else:
                    agent_df.to_csv(filename, mode='a', header=False, index=False)
                # Print stats for our agent
                print(f"Stats after iteration {i}")
                print(f"Number of publishers: {len(agent_df['publisher'].unique())}")
                print(f"Number of clicks at iteration {i}: {agent_df['num_clicks'].sum()}")
                print(f"Spent at iteration {i}: {agent_df['spent'].sum()}")
                # Save stats in Combinatorial LinUCB
                # Group by publisher if there are more than one iteration data
                grouped_data = agent_df.groupby(by=['publisher']) \
                    .sum() \
                    .reset_index() \
                    [['publisher', 'won_auctions', 'lost_auctions', 'num_clicks', 'spent']]
                grouped_data['cpc'] = grouped_data.apply(lambda row: row['spent'] / row['num_clicks'] if row['num_clicks'] > 0 else 0, axis=1)
                comb_linucb.save_stats(grouped_data)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, DoublyRobustBidder):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas)).detach().item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            agent.clear_utility()
            agent.clear_logs()

        auction.clear_revenue()
    # Save LinUCB parameters
    linucb_params = comb_linucb.linucb_params
    linucb_params['Run'] = run
    # Merge with previously saved parameters
    agent_df = pd.read_csv(filename, usecols=['publisher', 'num_clicks', 'Iteration', 'spent'])
    merged_df = pd.merge(
        agent_df,
        linucb_params,
        on=['publisher', 'Iteration'],
        how='outer'
    )
    merged_df.to_csv(f"{output_dir}/lin_ucb_params_run_{run}.csv", index=False)

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size, adv_embeddings, publisher_embeddings, knapsack_params = parse_config(args.config)
    # Instantiate knapsack parameters
    soglia_clicks = None
    soglia_spent = None
    soglia_cpc = None
    if 'soglia_clicks' in knapsack_params:
        soglia_clicks = knapsack_params['soglia_clicks']
    if 'soglia_spent' in knapsack_params:
        soglia_spent = knapsack_params['soglia_spent']
    if 'soglia_cpc' in knapsack_params:
        soglia_cpc = knapsack_params['soglia_cpc']

    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2allocation_regret = {}
    run2agent2estimation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}

    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2gamma = {}

    run2auction_revenue = {}

    # Repeated runs
    for run in range(num_runs):
        # Reinstantiate agents and auction per run
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)
        publishers = instantiate_publishers(publisher_embeddings)

        # Make sure we can write results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Placeholders for summary statistics per run
        agent2net_utility = defaultdict(list)
        agent2gross_utility = defaultdict(list)
        agent2allocation_regret = defaultdict(list)
        agent2estimation_regret = defaultdict(list)
        agent2overbid_regret = defaultdict(list)
        agent2underbid_regret = defaultdict(list)
        agent2best_expected_value = defaultdict(list)

        agent2CTR_RMSE = defaultdict(list)
        agent2CTR_bias = defaultdict(list)
        agent2gamma = defaultdict(list)

        auction_revenue = []

        start_run = time.time()
        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run()
        print(f'Run {run} took {time.time() - start_run} seconds')

        # Store
        run2agent2net_utility[run] = agent2net_utility
        run2agent2gross_utility[run] = agent2gross_utility
        run2agent2allocation_regret[run] = agent2allocation_regret
        run2agent2estimation_regret[run] = agent2estimation_regret
        run2agent2overbid_regret[run] = agent2overbid_regret
        run2agent2underbid_regret[run] = agent2underbid_regret
        run2agent2best_expected_value[run] = agent2best_expected_value

        run2agent2CTR_RMSE[run] = agent2CTR_RMSE
        run2agent2CTR_bias[run] = agent2CTR_bias
        run2agent2gamma[run] = agent2gamma

        run2auction_revenue[run] = auction_revenue

    def measure_per_agent2df(run2agent2measure, measure_name):
        df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
        for run, agent2measure in run2agent2measure.items():
            for agent, measures in agent2measure.items():
                for iteration, measure in enumerate(measures):
                    df_rows['Run'].append(run)
                    df_rows['Agent'].append(agent)
                    df_rows['Iteration'].append(iteration)
                    df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_per_agent(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure

        # fig, axes = plt.subplots(figsize=FIGSIZE)
        # plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        # min_measure, max_measure = 0.0, 0.0
        # sns.lineplot(data=df, x="Iteration", y=measure_name, hue="Agent", ax=axes)
        # plt.xticks(fontsize=FONTSIZE - 2)
        # plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        # if optimal is not None:
        #     plt.axhline(optimal, ls='--', color='gray', label='Optimal')
        #     min_measure = min(min_measure, optimal)
        # if log_y:
        #     plt.yscale('log')
        # if yrange is None:
        #     factor = 1.1 if min_measure < 0 else 0.9
        #     # plt.ylim(min_measure * factor, max_measure * 1.1)
        # else:
        #     plt.ylim(yrange[0], yrange[1])
        # plt.yticks(fontsize=FONTSIZE - 2)
        # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        # plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    net_utility_df = plot_measure_per_agent(run2agent2net_utility, 'Net Utility').sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    net_utility_df['Net Utility (Cumulative)'] = net_utility_df.groupby(['Agent', 'Run'])['Net Utility'].cumsum()
    plot_measure_per_agent(net_utility_df, 'Net Utility (Cumulative)')

    gross_utility_df = plot_measure_per_agent(run2agent2gross_utility, 'Gross Utility').sort_values(['Agent', 'Run', 'Iteration'])
    gross_utility_df.to_csv(f'{output_dir}/gross_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    gross_utility_df['Gross Utility (Cumulative)'] = gross_utility_df.groupby(['Agent', 'Run'])['Gross Utility'].cumsum()
    plot_measure_per_agent(gross_utility_df, 'Gross Utility (Cumulative)')

    # plot_measure_per_agent(run2agent2best_expected_value, 'Mean Expected Value for Top Ad')
    #
    # plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    # plot_measure_per_agent(run2agent2estimation_regret, 'Estimation Regret')
    # overbid_regret_df = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    # overbid_regret_df.to_csv(f'{output_dir}/overbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
    # underbid_regret_df = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    # underbid_regret_df.to_csv(f'{output_dir}/underbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
    #
    # plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE', log_y=True)
    # plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0) #, yrange=(.5, 5.0))
    #
    # shading_factor_df = plot_measure_per_agent(run2agent2gamma, 'Shading Factors')

    def measure2df(run2measure, measure_name):
        df_rows = {'Run': [], 'Iteration': [], measure_name: []}
        for run, measures in run2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_overall(run2measure, measure_name):
        # Generate DataFrame for Seaborn
        if type(run2measure) != pd.DataFrame:
            df = measure2df(run2measure, measure_name)
        else:
            df = run2measure
        # fig, axes = plt.subplots(figsize=FIGSIZE)
        # plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        # sns.lineplot(data=df, x="Iteration", y=measure_name, ax=axes)
        # min_measure = min(0.0, np.min(df[measure_name]))
        # max_measure = max(0.0, np.max(df[measure_name]))
        # plt.xlabel('Iteration', fontsize=FONTSIZE)
        # plt.xticks(fontsize=FONTSIZE - 2)
        # plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        # factor = 1.1 if min_measure < 0 else 0.9
        # plt.ylim(min_measure * factor, max_measure * 1.1)
        # plt.yticks(fontsize=FONTSIZE - 2)
        # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    auction_revenue_df = plot_measure_overall(run2auction_revenue, 'Auction Revenue')

    net_utility_df_overall = net_utility_df.groupby(['Run', 'Iteration'])['Net Utility'].sum().reset_index().rename(columns={'Net Utility': 'Social Surplus'})
    plot_measure_overall(net_utility_df_overall, 'Social Surplus')

    gross_utility_df_overall = gross_utility_df.groupby(['Run', 'Iteration'])['Gross Utility'].sum().reset_index().rename(columns={'Gross Utility': 'Social Welfare'})
    plot_measure_overall(gross_utility_df_overall, 'Social Welfare')

    auction_revenue_df['Measure Name'] = 'Auction Revenue'
    net_utility_df_overall['Measure Name'] = 'Social Surplus'
    gross_utility_df_overall['Measure Name'] = 'Social Welfare'

    columns = ['Run', 'Iteration', 'Measure', 'Measure Name']
    auction_revenue_df.columns = columns
    net_utility_df_overall.columns = columns
    gross_utility_df_overall.columns = columns

    pd.concat((auction_revenue_df, net_utility_df_overall, gross_utility_df_overall)).to_csv(f'{output_dir}/results_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
