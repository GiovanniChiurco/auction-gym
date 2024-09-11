from new_main import *
import multiprocessing
import argparse
from collections import defaultdict
import mlflow
import mlflow.sklearn
import os
import numpy as np


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


def measure_per_agent2df_one_run(agent2measure: defaultdict, measure_name, run):
    df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
    for agent in agent2measure:
        for iteration, measure in enumerate(agent2measure[agent]):
            df_rows['Run'].append(run)
            df_rows['Agent'].append(agent)
            df_rows['Iteration'].append(iteration)
            df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)


def run_iteration(auction, num_iter, publishers, agents2items):
    agent2net_utility = defaultdict(list)
    agent2gross_utility = defaultdict(list)

    auction_revenue = []

    for i in range(num_iter):
        # Compute the similarity between all users and agents
        noise_strength = 0.01
        all_user_context, all_users_agent_similarity = compute_all_users_agent_similarity(
            publishers, noise_strength, agents2items)
        # Create a mask for each agent
        mask_pub_agent = {}
        for publisher in publishers:
            mask_pub_agent[publisher.name] = np.zeros(publisher.num_auctions)
        # Variables to show progress bar
        pub_auctions = {publisher.name: publisher.num_auctions for publisher in publishers}
        total_auctions = sum(pub_auctions.values())
        with tqdm(total=total_auctions) as pbar:
            # Iteration over all auctions
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

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            # Log metrics to MLflow
            # mlflow.log_metrics({
            #     f"{agent.name}_net_utility": agent.net_utility,
            #     f"{agent.name}_gross_utility": agent.gross_utility,
            # }, step=i)

            agent.clear_utility()
            agent.clear_logs()

        auction_revenue.append(auction.revenue)
        # mlflow.log_metric("auction_revenue", auction.revenue, step=i)
        auction.clear_revenue()

    return agent2net_utility, agent2gross_utility, auction_revenue


def run_experiment(run, rng, agent_configs, agents2item_values, agents2items, config, max_slots, publisher_configs):
    print(f'Running experiment {run}')
    # Re-instantiate agents and auction per experiment
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items,
                                                                         agents2item_values, agents, max_slots)
    publishers = instantiate_publishers(rng, publisher_configs)

    # # Start MLflow run for each experiment
    # with mlflow.start_run(run_name=f"experiment_{run}"):
    #     mlflow.log_param("num_iter", num_iter)
    #     mlflow.log_param("num_agents", len(agents))
    #     mlflow.log_param("num_publishers", len(publishers))

    # Run simulation
    (agent2net_utility, agent2gross_utility, auction_revenue) = run_iteration(auction, num_iter, publishers, agents2items)

    print(f'Finished experiment {run}')

    net_utility_df = measure_per_agent2df_one_run(
        agent2net_utility,
        'Net Utility',
        run
    ).sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{num_iter}_iters_{run}_run.csv', index=False)

    # Store results in a dictionary
    return {
        'run': run,
        'agent2net_utility': agent2net_utility,
        'agent2gross_utility': agent2gross_utility,
        'auction_revenue': auction_revenue
    }

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    (rng, config, agent_configs, agents2items, agents2item_values, publisher_configs, num_runs, max_slots) = (
        parse_config(args.config))

    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment("nuo2")

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}

    run2auction_revenue = {}

    output_dir = config['output_dir']
    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parallelize the runs
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(run_experiment, [(run, rng, agent_configs, agents2item_values, agents2items, config,
                                                 max_slots, publisher_configs) for run in range(num_runs)])
        for result in results:
            run = result['run']
            run2agent2net_utility[run] = result['agent2net_utility']
            run2agent2gross_utility[run] = result['agent2gross_utility']
            run2auction_revenue[run] = result['auction_revenue']

    num_iter = config['num_iter']
    net_utility_df = measure_per_agent2df(
        run2agent2net_utility,
        'Net Utility').sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{num_iter}_iters_{num_runs}_runs.csv', index=False)
