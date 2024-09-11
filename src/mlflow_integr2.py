from new_main import *
import argparse
import mlflow
import mlflow.sklearn
import multiprocessing
import os
from collections import defaultdict
import numpy as np
import time


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


def run_iteration(auction, num_iter, publishers, agents2items, run):
    agent2net_utility = defaultdict(list)
    agent2gross_utility = defaultdict(list)

    auction_revenue = []

    for i in range(num_iter):

        gen_auction_time = time.time()

        # Simula opportunità e aggiorna metriche
        noise_strength = 0.01
        all_user_context, all_users_agent_similarity = compute_all_users_agent_similarity(
            publishers, noise_strength, agents2items)

        print(f'[Run {run} - Iteration {i}]: Generated auction in {time.time() - gen_auction_time} seconds')

        mask_pub_agent = {publisher.name: np.zeros(publisher.num_auctions) for publisher in publishers}

        simulated_auctions_time = time.time()

        while not all(np.all(mask == 1) for mask in mask_pub_agent.values()):
            publisher = np.random.choice(publishers)
            mask = mask_pub_agent[publisher.name]

            try:
                idx = np.where(mask == 0)[0][0]
            except IndexError:
                continue

            curr_user_context = all_user_context[publisher.name][idx]
            auction.simulate_opportunity(all_users_agent_similarity, publisher.name, idx, curr_user_context)
            mask[idx] = 1

        print(f'[Run {run} - Iteration {i}]: Simulated auctions in {time.time() - simulated_auctions_time} seconds')

        for agent in auction.agents:

            update_time = time.time()

            agent.update(iteration=i)

            print(f'[Run {run} - Iteration {i}]: Updated agent {agent.name} in {time.time() - update_time} seconds')

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            # Loggare la metrica di net utility per ogni agente su MLflow
            mlflow.log_metric(f"{agent.name}_net_utility", agent.net_utility, step=i)

            agent.clear_utility()
            agent.clear_logs()

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()

        print(f'[Run {run} - Iteration {i}]: Finished iteration {i} in {time.time() - gen_auction_time} seconds')

    return agent2net_utility, agent2gross_utility, auction_revenue


def run_experiment(run, rng, agent_configs, agents2item_values, agents2items, config, max_slots, publisher_configs,
                   parent_run_id):
    # Creazione di una nuova run nidificata all'interno dell'esperimento principale
    with mlflow.start_run(run_name=f"Run_{run}", nested=True, parent_run_id=parent_run_id):
        print(f'Running experiment {run}')
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, _, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents,
                                                               max_slots)
        publishers = instantiate_publishers(rng, publisher_configs)

        curr_iter_time = time.time()

        agent2net_utility, agent2gross_utility, auction_revenue = run_iteration(auction, num_iter, publishers,
                                                                                agents2items, run)

        print(f'Finished experiment {run} in {time.time() - curr_iter_time} seconds')

        # Salvare i risultati su file
        net_utility_df = measure_per_agent2df_one_run(
            agent2net_utility, 'Net Utility', run).sort_values(['Agent', 'Run', 'Iteration'])
        net_utility_df.to_csv(f'{output_dir}/net_utility_{num_iter}_iters_{run}_run.csv', index=False)

        return {'run': run, 'agent2net_utility': agent2net_utility, 'agent2gross_utility': agent2gross_utility,
                'auction_revenue': auction_revenue}


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    (rng, config, agent_configs, agents2items, agents2item_values, publisher_configs, num_runs, max_slots) = (
        parse_config(args.config))

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}

    run2auction_revenue = {}

    output_dir = config['output_dir']
    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iniziare l'esperimento principale e ottenere il suo ID per i sotto-esperimenti
    with mlflow.start_run(run_name=f"experiment_{time.time()}") as main_run:
        parent_run_id = main_run.info.run_id

        output_dir = config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        run2agent2net_utility = {}
        run2agent2gross_utility = {}
        run2auction_revenue = {}

        # Parallelizzare le simulazioni
        with multiprocessing.Pool(processes=3) as pool:
            results = pool.starmap(run_experiment, [(run, rng, agent_configs, agents2item_values, agents2items, config,
                                                     max_slots, publisher_configs, parent_run_id) for run in
                                                    range(num_runs)])

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
