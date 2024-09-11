import argparse
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import time

from Agent import Agent
from AuctionAllocation import * # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  #  LogisticTSAllocator, OracleAllocator

from Publisher import Publisher


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

    # Read ads embeddings
    ads_embeddings = pickle.load(open('src/publisher_embedding/data/nuova_descr_annunci_embed.pkl', 'rb'))
    # Pick embeddings from a pre-defined set
    agents2items = {
        agent_config['name']: ads_embeddings[agent_config['product_name']]
        for agent_config in agent_configs
    }

    agents2item_values = {
        #agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
        agent_config['name']: np.ones(agent_config['num_items'])
        for agent_config in agent_configs
    }

    # Publishers configs
    publisher_configs = config['publishers']
    # Get random publishers at the beginning of the simulation
    # To change publishers at each iteration, put this line inside the loop
    # publisher_configs = get_random_publishers(num_publishers, num_auctions)

    return (rng, config, agent_configs, agents2items, agents2item_values, publisher_configs,
            num_runs, max_slots)


def instantiate_agents(
        rng, agent_configs, agents2item_values, agents2items,
):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              item_values=agents2item_values[agent_config['name']],
              budget=agent_config['budget'],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              memory=(0 if 'memory' not in agent_config.keys() else agent_config['memory']))
        for agent_config in agent_configs
    ]

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

    return agents


def instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots):
    return (Auction(rng,
                    eval(f"{config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    config['num_participants_per_round']),
            config['num_iter'], config['rounds_per_iter'], config['output_dir'])


def instantiate_publishers(rng, publisher_configs):
    return [
        Publisher(
            rng=rng,
            name=publisher_config['name'],
            num_auctions=publisher_config['num_auctions'],
        )
        for publisher_config in publisher_configs
    ]


def logistic_fun(x, k=1, x0=0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def add_noise(user_embedding, noise_strength):
    noise = np.random.normal(0, noise_strength, user_embedding.shape)
    return user_embedding + noise


def compute_user_embed_matrix(publisher, noise_strength):
    user_embed_matrix = np.zeros((publisher.num_auctions, publisher.embedding.size))
    for i in range(publisher.num_auctions):
        user_embed_matrix[i] = add_noise(publisher.embedding, noise_strength)
    return user_embed_matrix


def compute_all_users_matrix(publishers, noise_strength):
    all_users_matrix = {}
    for publisher in publishers:
        all_users_matrix[publisher.name] = compute_user_embed_matrix(publisher, noise_strength)
    return all_users_matrix


def agent_pub_user_similarity(agents2items, publisher, all_users_matrix):
    user_pub_sim = {}
    curr_pub_matrix = all_users_matrix[publisher.name]
    for agent, ad_embedding in agents2items.items():
        sim = cosine_similarity(ad_embedding.reshape(1, -1), curr_pub_matrix)
        user_pub_sim[agent] = sim
    return user_pub_sim


def compute_matrix_user_ad(agents2items, publishers, all_users_matrix):
    matrix_user_ad = {}
    for publisher in publishers:
        matrix_user_ad[publisher.name] = agent_pub_user_similarity(agents2items, publisher, all_users_matrix)
    return matrix_user_ad


def compute_all_users_agent_similarity(publishers, noise_strength, agents2items):
    all_users_matrix = compute_all_users_matrix(publishers, noise_strength)
    similarity_matrix = compute_matrix_user_ad(agents2items, publishers, all_users_matrix)
    return all_users_matrix, similarity_matrix


def get_random_publishers(num_publishers, num_auctions):
    sites_dir = "src/publisher_embedding/data/sites/"
    sites = [site.replace(".pkl", "") for site in os.listdir(sites_dir)]
    chosen_site_list = np.random.choice(sites, num_publishers, replace=False)
    return [
        {
            "name": site,
            "num_auctions": num_auctions
        }
        for site in chosen_site_list
    ]


def new_simulation_run():
    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')
        # Compute the similarity between all users and agents
        noise_strength = 0.01
        all_user_context, all_users_agent_similarity = compute_all_users_agent_similarity(publishers, noise_strength, agents2items)
        # Create a mask for each agent
        mask_pub_agent = {}
        for publisher in publishers:
            mask_pub_agent[publisher.name] = np.zeros(publisher.num_auctions)
        # Iteration
        # Variables to show progress bar
        pub_auctions = {publisher.name: publisher.num_auctions for publisher in publishers}
        total_auctions = sum(pub_auctions.values())

        start_time = time.time()

        with tqdm(total=total_auctions) as pbar:
            while not all(np.all(mask == 1) for mask in mask_pub_agent.values()):
                publisher = np.random.choice(publishers)
                mask = mask_pub_agent[publisher.name]
                # Catch the case when all auctions have been simulated for the current publisher
                try:
                    idx = np.where(mask == 0)[0][0]
                except IndexError:
                    continue

                curr_user_context = all_user_context[publisher.name][idx]
                auction.simulate_opportunity(all_users_agent_similarity, publisher.name, idx, curr_user_context)

                mask[idx] = 1
                pbar.update(1)

        # with tqdm(total=total_auctions) as pbar:
        #     while sum(pub_auctions.values()) > 0:
        #         for publisher in publishers:
        #             if pub_auctions[publisher.name] > 0:
        #                 # Generate a user from the current publisher
        #                 curr_user_context = publisher.generate_user_context()
        #                 # Simulate the auction
        #                 auction.simulate_opportunity(curr_user_context, publisher.name)
        #
        #                 pub_auctions[publisher.name] -= 1
        #                 pbar.update(1)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f'Time taken for {total_auctions} auctions to complete: {elapsed_time:.2f} seconds')

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]

        result = pd.DataFrame({'Name': names, 'Net': net_utilities, 'Gross': gross_utilities})

        print(result)
        print(f'\tAuction revenue: \t {auction.revenue}')

        num_auctions_won = 0
        for agent_id, agent in enumerate(auction.agents):
            print(f'Agent {agent.name} has spent {agent.get_agent_spent()}, '
                  f'winning {agent.get_agent_opp_won()} opportunities (win ratio: {agent.get_agent_won_ratio()})')
            print(f'Mean value paid for a won bid: {agent.get_mean_won_bid()}')

            num_auctions_won += agent.get_agent_opp_won()

            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

            agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, DoublyRobustBidder):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas)).detach().item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            agent2best_expected_value[agent.name].append(best_expected_value)

            # Save bids info for each agent in a csv file
            # agent_bids_info = agent.get_agent_bid_df()
            # agent_bids_info.to_csv(f'{output_dir}/{agent.name}_bids_info_{num_iter}_iters_{num_runs}_runs.csv', index=False)
            print(agent.agent_stats_won_publisher())

            print('Average Best Value for Agent: ', best_expected_value)
            agent.clear_utility()
            agent.clear_logs()

        print(f'Total number of auctions won: {num_auctions_won}')

        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time - start_time
        print(f'Iteration {i} completed in {iter_elapsed_time:.2f} seconds')

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()


def simulation_run():
    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')

        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]

        result = pd.DataFrame({'Name': names, 'Net': net_utilities, 'Gross': gross_utilities})

        print(result)
        print(f'\tAuction revenue: \t {auction.revenue}')

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

            agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, DoublyRobustBidder):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas)).detach().item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            agent2best_expected_value[agent.name].append(best_expected_value)

            print('Average Best Value for Agent: ', best_expected_value)
            agent.clear_utility()
            agent.clear_logs()

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    (rng, config, agent_configs, agents2items, agents2item_values, publisher_configs, num_runs, max_slots) = (
        parse_config(args.config))

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
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items,
                                                                             agents2item_values, agents, max_slots)
        publishers = instantiate_publishers(rng, publisher_configs)

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

        start_time = time.time()

        # Run simulation (with global parameters -- fine for the purposes of this script)
        # simulation_run()
        new_simulation_run()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Run {run} completed in {elapsed_time:.2f} seconds')

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

    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.lineplot(data=df, x="Iteration", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.pdf", bbox_inches='tight')
        # plt.show()
        return df

    net_utility_df = plot_measure_per_agent(run2agent2net_utility, 'Net Utility').sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.csv', index=False)

    net_utility_df['Net Utility (Cumulative)'] = net_utility_df.groupby(['Agent', 'Run'])['Net Utility'].cumsum()
    plot_measure_per_agent(net_utility_df, 'Net Utility (Cumulative)')

    gross_utility_df = plot_measure_per_agent(run2agent2gross_utility, 'Gross Utility').sort_values(['Agent', 'Run', 'Iteration'])
    gross_utility_df.to_csv(f'{output_dir}/gross_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.csv', index=False)

    gross_utility_df['Gross Utility (Cumulative)'] = gross_utility_df.groupby(['Agent', 'Run'])['Gross Utility'].cumsum()
    plot_measure_per_agent(gross_utility_df, 'Gross Utility (Cumulative)')

    plot_measure_per_agent(run2agent2best_expected_value, 'Mean Expected Value for Top Ad')

    plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    plot_measure_per_agent(run2agent2estimation_regret, 'Estimation Regret')
    overbid_regret_df = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/overbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.csv', index=False)
    underbid_regret_df = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/underbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.csv', index=False)

    plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE', log_y=True)
    plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0) #, yrange=(.5, 5.0))

    shading_factor_df = plot_measure_per_agent(run2agent2gamma, 'Shading Factors')

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
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        sns.lineplot(data=df, x="Iteration", y=measure_name, ax=axes)
        min_measure = min(0.0, np.min(df[measure_name]))
        max_measure = max(0.0, np.max(df[measure_name]))
        plt.xlabel('Iteration', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        factor = 1.1 if min_measure < 0 else 0.9
        plt.ylim(min_measure * factor, max_measure * 1.1)
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.pdf", bbox_inches='tight')
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

    pd.concat((auction_revenue_df, net_utility_df_overall, gross_utility_df_overall)).to_csv(f'{output_dir}/results_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.csv', index=False)
