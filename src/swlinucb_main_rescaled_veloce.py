import argparse
import multiprocessing
import os

import pandas as pd
from SWCLinUCB import SWCombinatorialLinUCBOpt
from new_main import *
import time
import pickle
import time

def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    random_seed = config['random_seed']
    rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)

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

    adv_embedding_path = config['adv_embedding_path']
    adv_embeddings = pickle.load(open(adv_embedding_path, 'rb'))
    # Adv embeddings for agents
    agents2items = {
        agent_config['name']: adv_embeddings[agent_config['adv_name']]
        for agent_config in agent_configs
    }
    # Adv values for agents equal to 1.0 for all agents and advs
    agents2item_values = {
        agent_config['name']: np.array([1.0] * agent_config['num_items'], dtype=np.float32)
        for agent_config in agent_configs
    }
    publisher_embeddings_path = config['publisher_embedding_path']
    publisher_embeddings = pickle.load(open(publisher_embeddings_path, 'rb'))
    # Rescaled publisher embeddings
    rescaled_publisher_embeddings_path = config['rescaled_publisher_embedding_path']
    rescaled_publisher_embeddings = pickle.load(open(rescaled_publisher_embeddings_path, 'rb'))
    # Window size for SWCombinatorialLinUCB
    window_size_list = config['window_size_list']

    return (rng, config, random_seed, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size,
            embedding_var, obs_embedding_size, adv_embeddings, publisher_embeddings, rescaled_publisher_embeddings, window_size_list)

def get_partecipant_mask(A, num_participants_per_round, rng):
    # Dimensioni
    n, m, p, r = A.shape
    # Numero totale di elementi nelle dimensioni (n, m, p)
    total_positions = n * m * p

    # Genera numeri casuali per ordinamento
    random_numbers = rng.random((total_positions, r))

    # Ordina i numeri casuali lungo l'ultima dimensione
    sorted_indices = np.argsort(random_numbers, axis=1)

    # Prendi i primi num_participants_per_round indici per ciascun gruppo
    selected_indices = sorted_indices[:, :num_participants_per_round]

    # Prepara maschera finale
    mask = np.zeros((n, m, p, r), dtype=int)

    # Crea array di indici per le prime tre dimensioni
    i_indices, j_indices, k_indices = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )

    # Flatten per allineare agli indici selezionati
    i_indices = i_indices.ravel()
    j_indices = j_indices.ravel()
    k_indices = k_indices.ravel()

    # Assegna i valori 1 alla maschera usando gli indici selezionati
    for idx in range(total_positions):
        mask[i_indices[idx], j_indices[idx], k_indices[idx], selected_indices[idx]] = 1

    return mask

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
    df_before_drift = df[['publisher', 'true_clicks_before_drift', 'clicks_before_drift', 'impressions_before_drift']]
    df_before_drift = df_before_drift.rename(columns={'true_clicks_before_drift': 'true_clicks', 'clicks_before_drift': 'clicks', 'impressions_before_drift': 'impressions'})
    n, clicks, impressions = get_data(df_before_drift)
    df_before_drift = solver(df_before_drift, n, clicks, impressions, soglia_ctr)
    df_after_drift = df[['publisher', 'true_clicks_after_drift', 'clicks_after_drift', 'impressions_after_drift']]
    df_after_drift = df_after_drift.rename(columns={'true_clicks_after_drift': 'true_clicks', 'clicks_after_drift': 'clicks', 'impressions_after_drift': 'impressions'})
    n, clicks, impressions = get_data(df_after_drift)
    df_after_drift = solver(df_after_drift, n, clicks, impressions, soglia_ctr)
    return df_before_drift, df_after_drift

def simulate_auctions(
    publisher_embeddings: dict,
    adv_embeddings: dict,
    pub_list: list,
    num_iter: int,
    rounds_per_iter: int,
    num_participants_per_round: int,
    iteration_drift: int = 30,
    noise_std: float = 0.01,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()
        
    # Filter publishers
    t0 = time.time()
    
    selected_publishers_embeddings = {pub: publisher_embeddings[pub] for pub in pub_list}
    
    print(f"Filtering publishers took: {time.time() - t0:.4f} seconds")
    
    # Convert to array the dicts
    t1_a = time.time()
    
    publisher_embeddings_array = np.array(list(selected_publishers_embeddings.values()))
    
    print(f"Converting arrays took: {time.time() - t1_a:.4f} seconds")
    t1_b = time.time()
    
    publisher_embeddings_array_tiled = np.tile(publisher_embeddings_array, 
                                             (num_iter, rounds_per_iter, 1, 1))
    
    print(f"Tiling arrays took: {time.time() - t1_b:.4f} seconds")
    t1_c = time.time()

    # Create and add noise

    publisher_embeddings_array_tiled = rng.normal(
        loc=publisher_embeddings_array_tiled,  # usa l'array esistente come media
        scale=noise_std,                        
        size=None,                             # usa la dimensione dell'array esistente
    )

    print(f"Generating an adding noise took: {time.time() - t1_c:.4f} seconds")
    # Convert advertiser embeddings and compute scalar products
    t2 = time.time()
    
    adv_embeddings_array = np.array(list(adv_embeddings.values()))
    
    # Reshape for better cache utilization
    pub_shape = publisher_embeddings_array_tiled.shape
    publisher_embeddings_array_tiled = publisher_embeddings_array_tiled.reshape(-1, pub_shape[-1])
    scalar_products = np.dot(publisher_embeddings_array_tiled, adv_embeddings_array.T)
    scalar_products = scalar_products.reshape(pub_shape[0], pub_shape[1], pub_shape[2], -1)

    print(f"Computing scalar products took: {time.time() - t2:.4f} seconds")
    
    # Compute sigmoids
    t3 = time.time()
    
    mean_scores = np.mean(scalar_products)
    std_scores = np.std(scalar_products)
    sigmoids = 1 / (1 + np.exp(-(scalar_products - mean_scores) / (0.5 * std_scores)))
    
    print(f"Computing sigmoids took: {time.time() - t3:.4f} seconds")
    
    # Get participant mask
    t4_a = time.time()

    partecipant_mask = get_partecipant_mask(sigmoids, num_participants_per_round, rng)
    
    print(f"Getting participant mask took: {time.time() - t4_a:.4f} seconds")
    
    t4_b = time.time()
    
    partecipant_sigmoids = sigmoids * partecipant_mask
    
    print(f"Multiplying sigmoids took: {time.time() - t4_b:.4f} seconds")

    # From the 30-th iteration, our bids are the 80% of the actual ones
    # The CTR remains the same to determine click/no-click
    partecipant_sigmoids[iteration_drift:,:,:,0] = partecipant_sigmoids[iteration_drift:,:,:,0] * 0.8

    # Determine winners and calculate CTR/impressions
    t5 = time.time()
    
    winners = np.argmax(partecipant_sigmoids, axis=3)
    our_sigmoids = sigmoids[:,:,:,0]
    our_ctr = np.where((winners==0), our_sigmoids, 0)
    our_clicks = rng.binomial(1, our_ctr).sum(axis=1)
    our_impressions = (winners==0).astype(int)
    
    print(f"Determining winners and calculating CTR took: {time.time() - t5:.4f} seconds")
    
    # Aggregate results over the rounds of each iteration
    clicks = our_ctr.sum(axis=1)
    impressions = our_impressions.sum(axis=1)
    
    results = pd.DataFrame()
    for i in range(num_iter):
        curr_results = pd.DataFrame({
            'publisher': pub_list,
            'Iteration': i,
            'true_clicks': clicks[i],
            'clicks': our_clicks[i],
            'impressions': impressions[i]
        })
        results = pd.concat([results, curr_results])
    
    # Aggregate results before and after the drift
    our_clicks_before_drift = our_clicks[:iteration_drift,:].mean(axis=0)
    our_clicks_after_drift = our_clicks[iteration_drift:,:].mean(axis=0)
    our_impressions_before_drift = impressions[:iteration_drift,:].mean(axis=0)
    our_impressions_after_drift = impressions[iteration_drift:,:].mean(axis=0)
    true_clicks_before_drift = clicks[:iteration_drift,:].mean(axis=0)
    true_clicks_after_drift = clicks[iteration_drift:,:].mean(axis=0)

    group_pub_res = pd.DataFrame({
        'publisher': pub_list,
        'iteration_drift': [iteration_drift] * len(pub_list),
        'true_clicks_before_drift': true_clicks_before_drift,
        'true_clicks_after_drift': true_clicks_after_drift,
        'clicks_before_drift': our_clicks_before_drift,
        'clicks_after_drift': our_clicks_after_drift,
        'impressions_before_drift': our_impressions_before_drift,
        'impressions_after_drift': our_impressions_after_drift
    })

    return results, group_pub_res

def simulation_run(
        run: int, init_publisher_list: list[Publisher], init_publisher_embeddings: dict, sim_auctions: pd.DataFrame, num_iter: int,
        soglia_ctr: float, embedding_size: int, alpha: float, window_size: int
) -> tuple[pd.DataFrame]:
    agent_stats = pd.DataFrame()
    comb_linucb = SWCombinatorialLinUCBOpt(
        alpha=alpha, d=embedding_size, publisher_list=init_publisher_list, window_size=window_size
    )
    for i in range(num_iter):
        if i > 1:
            publisher_list = comb_linucb.round_iteration(
                curr_publisher_list=publisher_list,
                run=run,
                iteration=i,
                soglia_ctr=soglia_ctr
            )
        else:
            publisher_list = init_publisher_list
            comb_linucb.curr_superarm[i] = publisher_list

        publisher_name_list = [publisher.name for publisher in publisher_list]
        curr_iter_df = sim_auctions[(sim_auctions['Iteration'] == i)&(sim_auctions['publisher'].isin(publisher_name_list))]
        agent_stats_pub = curr_iter_df.to_dict(orient='records')

        group_iter = curr_iter_df.groupby('Iteration').agg({'clicks': 'sum', 'impressions': 'sum'}).reset_index()
        group_iter['CTR'] = group_iter['clicks'] / group_iter['impressions']
        print(f'[Run {run}, Iteration {i}] Actual CTR: {group_iter["CTR"].values[0]}')
        
        comb_linucb.update(agent_stats_pub, init_publisher_embeddings)

        agent_stats = pd.concat([agent_stats, curr_iter_df])

    return agent_stats


def run_simulation(
        output_dir: str, run: int, random_seed: int, init_publisher_list: list[Publisher], publisher_embeddings: dict, num_iter: int, rounds_per_iter: int, 
        soglia_ctr: float, embedding_size: int, adv_embeddings: dict, alpha: float, rng: np.random.Generator = None, iteration_drift: int = 30, num_participants_per_round: int = 4, window_size: int = 10):

    init_publisher_embeddings = {publisher.name: publisher_embeddings[publisher.name] for publisher in init_publisher_list}

    # Set up Random Number Generator
    # Different seed for each run
    rng = np.random.default_rng(run+random_seed)
    np.random.seed(run+random_seed)

    sim_auctions, group_pub_res = simulate_auctions(
        publisher_embeddings=init_publisher_embeddings,
        adv_embeddings=adv_embeddings,
        pub_list=[publisher.name for publisher in init_publisher_list],
        num_iter=num_iter,
        rounds_per_iter=rounds_per_iter,
        iteration_drift=iteration_drift,
        num_participants_per_round=num_participants_per_round,
        noise_std=0.01,
        rng=rng
    )
    sim_auctions.to_csv(
        os.path.join(output_dir, f'sim_auctions_run_{run}.csv'), index=False)
    
    opt_exp_results_before_drift, opt_exp_results_after_drift = knapsack(group_pub_res, soglia_ctr=soglia_ctr)
    opt_exp_results_before_drift.to_csv(
        os.path.join(output_dir, f'opt_exp_results_before_drift_run_{run}.csv'), index=False)
    opt_exp_results_after_drift.to_csv(
        os.path.join(output_dir, f'opt_exp_results_after_drift_run_{run}.csv'), index=False)

    rescaled_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}

    agent_stats = simulation_run(run, init_publisher_list, rescaled_publisher_embeddings, sim_auctions, num_iter, soglia_ctr, embedding_size, alpha, window_size)

    agent_stats.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}_ws_{window_size}.csv'), index=False)


if __name__ == "__main__":
    """This script runs the simulation for the Combinatorial LinUCB algorithm with the given configuration file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    (rng, config, random_seed, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var,
     obs_embedding_size, adv_embeddings, publisher_embeddings, rescaled_publisher_embeddings, window_size_list) = parse_config(args.config)
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values,
                                                                         agents, max_slots, embedding_size,
                                                                         embedding_var, obs_embedding_size)
    publishers = instantiate_publishers(rescaled_publisher_embeddings, rounds_per_iter)

    num_participants_per_round = config['num_participants_per_round']
    iteration_drift = config['iteration_drift']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rng.shuffle(publishers)
    
    num_pub = 300
    init_publisher_list = publishers[:num_pub]
    # Exclude the following publishers such that we always have publishers with at least 1 impression
    pub_to_exclude = ['dolcipassioni.net', 'healthy.thewom.it', 'unita.it', 'disboard.org', 'ilclubdellericette.it', 
                      'agrodolce.it', 'hovogliadidolce.it', 'giallozafferano.it', 'recetasgratis.net', 'prodottitipicitoscani.it', 
                      'wiadomosci.onet.pl', 'approdocalabria.it', 'buttalapasta.it']
    init_publisher_list = [pub for pub in init_publisher_list if pub.name not in pub_to_exclude]
    
    soglia_ctr_list = [0.65]
    alpha_list = [1]
    
    tasks = []
    for soglia_ctr in soglia_ctr_list:
        for window_size in window_size_list:
            for alpha in alpha_list:
                for run in range(num_runs):
                    tasks.append((output_dir, run, random_seed, init_publisher_list, publisher_embeddings, num_iter, rounds_per_iter, soglia_ctr, embedding_size, adv_embeddings, alpha, rng, iteration_drift, num_participants_per_round, window_size))

    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
