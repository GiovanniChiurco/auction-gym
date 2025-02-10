import argparse
import multiprocessing
import os
import time

import pandas as pd
from CUCB import CUCB
from CUCBNuo import CUCBNuo
from new_main import *

from ortools.linear_solver import pywraplp


def get_data(
        df: pd.DataFrame,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = df.shape[0]
    clicks = df['clicks'].values
    impressions = df['impressions'].values
    return n, clicks, impressions

def solver(
        df: pd.DataFrame,
        n: int,
        clicks: np.ndarray,
        impressions: np.ndarray,
        soglia_ctr: float = None,
) -> pd.DataFrame:
    solver = pywraplp.Solver.CreateSolver('SCIP')
    # Boolean variables
    x = [solver.BoolVar(f'x{i}') for i in range(n)]
    x_np = np.array(x)
    # Objective function
    solver.Maximize(np.dot(clicks, x_np))
    if soglia_ctr is not None:
        # CTR constraint
        solver.Add(np.dot(clicks, x_np) >= soglia_ctr * np.dot(impressions, x_np))
    # Solve the knapsack problem
    status = solver.Solve()
    results = pd.DataFrame(columns=df.columns)
    # Cycle over the boolean variables to get the selected rows
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(n):
            if x[i].solution_value() == 1:
                if results.empty:
                    results = df.iloc[[i]]
                else:
                    results = pd.concat([results, df.iloc[[i]]])
        print("Knapsack Solver: Optimal solution!")
        print(f"Total clicks = {results['clicks'].sum()}")
        print(f"Total impressions = {results['impressions'].sum()}")
        print(f"Number of selected publishers = {results.shape[0]}")
        if soglia_ctr is not None:
            if results['impressions'].sum() != 0:
                print(f"CTR = {results['clicks'].sum() / results['impressions'].sum()}")
            else:
                print("CTR = undefined (division by zero)")
    else:
        print("Knapsack Solver: No feasible solution found.")
        # Return empty dataframe
        return pd.DataFrame()
    return results

def knapsack(
        df: pd.DataFrame,
        soglia_ctr: float = None,
) -> pd.DataFrame:
    n, clicks, impressions = get_data(df)
    return solver(df, n, clicks, impressions, soglia_ctr)


def simulation_run(
        run: int, init_publisher_list: list[Publisher], sim_auctions: pd.DataFrame, num_iter: int,
        rounds_per_iter: int, soglia_ctr: float, alpha: float
):
    agent_stats = pd.DataFrame()
    cucb = CUCBNuo(publisher_list=init_publisher_list, alpha=alpha)
    for i in range(num_iter):
        if i > 0:
            publisher_list = cucb.round_iteration(
                curr_publisher_list=publisher_list,
                run=run,
                iteration=i,
                soglia_ctr=soglia_ctr
            )
        else:
            cucb.set_time_t(i+1)
            publisher_list = init_publisher_list

        publisher_name_list = [publisher.name for publisher in publisher_list]
        curr_iter_df = sim_auctions[(sim_auctions['Iteration'] == i)&(sim_auctions['publisher'].isin(publisher_name_list))]
        agent_stats_pub = curr_iter_df.to_dict(orient='records')

        group_iter = curr_iter_df.groupby('Iteration').agg({'clicks': 'sum', 'impressions': 'sum'}).reset_index()
        group_iter['CTR'] = group_iter['clicks'] / group_iter['impressions']
        print(f'[Run {run}, Iteration {i}] Actual CTR: {group_iter["CTR"].values[0]}')
        
        for publisher_data in agent_stats_pub:
            cucb.update_arm(
                publisher_name=publisher_data['publisher'],
                clicks=publisher_data['clicks'],
                impressions=publisher_data['impressions']
            )

        agent_stats = pd.concat([agent_stats, curr_iter_df])
    return agent_stats


def run_simulation(
        output_dir: str, run: int, random_seed: int, init_publisher_list: list[Publisher], auction: Auction, num_iter: int, rounds_per_iter: int, 
        soglia_ctr: float, alpha: float, embedding_size: int, adv_embeddings: dict, rng: np.random.Generator):
    # Set up Random Number Generator
    # Different seed for each run
    rng = np.random.default_rng(run+random_seed)
    np.random.seed(run+random_seed)

    # Read the simulation data
    sim_auctions = pd.read_csv(os.path.join(output_dir, f'sim_auctions_run_{run}.csv'))
    group_pub_res = pd.read_csv(os.path.join(output_dir, f'group_pub_res_run_{run}.csv'))

    if not os.path.exists(os.path.join(output_dir, f'opt_exp_results_run_{run}.csv')):
        # Run the knapsack algorithm
        opt_exp_results = knapsack(group_pub_res, soglia_ctr=soglia_ctr)
        opt_exp_results.to_csv(os.path.join(output_dir, f'opt_exp_results_run_{run}.csv'), index=False)

    agent_stats = simulation_run(run, init_publisher_list, sim_auctions, num_iter, rounds_per_iter, soglia_ctr, alpha)

    agent_stats.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.csv'), index=False)



if __name__ == "__main__":
    """This script runs the simulation for the CUCB algorithm with the given configuration file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    (rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var,
     obs_embedding_size, adv_embeddings, publisher_embeddings, knapsack_params) = parse_config(args.config)
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values,
                                                                         agents, max_slots, embedding_size,
                                                                         embedding_var, obs_embedding_size)
    publishers = instantiate_publishers(publisher_embeddings, rounds_per_iter)

    random_seed = config['random_seed']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rng.shuffle(publishers)
    init_publisher_list = publishers[:300]
    # Exclude the following publishers such that we always have publishers with at least 1 impression
    pub_to_exclude = ['dolcipassioni.net', 'healthy.thewom.it', 'unita.it', 'disboard.org', 'ilclubdellericette.it', 
                      'agrodolce.it', 'hovogliadidolce.it', 'giallozafferano.it', 'recetasgratis.net', 'prodottitipicitoscani.it', 
                      'wiadomosci.onet.pl','approdocalabria.it', 'buttalapasta.it'
                      ]
    init_publisher_list = [pub for pub in init_publisher_list if pub.name not in pub_to_exclude]

    # Hyperparameters
    alpha_list = [1]
    soglia_ctr_list = [0.86]

    tasks = []
    for alpha in alpha_list:
        for soglia_ctr in soglia_ctr_list:
            for run in range(num_runs):
                tasks.append((output_dir, run, random_seed, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, alpha, embedding_size, adv_embeddings, rng))

    start_time = time.time()
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
