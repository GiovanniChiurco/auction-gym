import multiprocessing
from CombinatorialLinUCB_nuo import CombinatorialLinUCBNuo
from SW_CLinUCB import SWCLinUCB
from new_main import *
import time
import pickle
import re


def read_results(result_dir: str) -> pd.DataFrame:
    # Leggo i file
    files_to_read = [file for file in os.listdir(result_dir) if file.endswith('.csv')]
    pattern = r'agent_stats_run_(\d+)_ctr_(\d+\.?\d*)_alpha_(\d+\.?\d*)'
    files_per_alpha = {}
    for file in files_to_read:
        match = re.search(pattern, file)
        if match:
            alpha = float(match.group(3))
            if alpha not in files_per_alpha:
                files_per_alpha[alpha] = [file]
            else:
                files_per_alpha[alpha].append(file)
    # Carico i risultati su un unico dataframe aggiungendo la colonna alpha
    results = pd.DataFrame()
    for alpha, files in files_per_alpha.items():
        for file in files:
            curr_results = pd.read_csv(result_dir + file)
            curr_results['alpha'] = alpha
            results = pd.concat([results, curr_results])
    # Prima raggruppo per Run e Iteration per avere il dato aggregato per ogni iterazione
    grouped_results_run_iter = results.groupby(['alpha', 'Run', 'Iteration']) \
        .agg({'clicks': 'sum', 'impressions': 'sum', 'true_clicks': 'sum'}) \
        .reset_index()
    # Medio sulle run
    grouped_results = grouped_results_run_iter.groupby(['alpha', 'Iteration']).mean().reset_index()
    # Calcolo CTR e true CTR
    grouped_results['ctr'] = grouped_results['clicks'] / grouped_results['impressions']
    grouped_results['true_ctr'] = grouped_results['true_clicks'] / grouped_results['impressions']

    return grouped_results

def simulate_auctions_random(
        publisher_list: List[Publisher], user_contexts: dict, sigmoids:dict, auction: Auction, iteration: int
):
    # Create a mask for each agent
    mask_pub_agent = {}
    for publisher in publisher_list:
        mask_pub_agent[publisher.name] = np.zeros(publisher.num_auctions)
    while not all(np.all(mask == 1) for mask in mask_pub_agent.values()):
        publisher = np.random.choice(publisher_list)
        mask = mask_pub_agent[publisher.name]
        # Catch the case when all auctions have been simulated for the current publisher
        try:
            idx = np.where(mask == 0)[0][0]
        except IndexError:
            continue

        curr_user_context = user_contexts[publisher.name][iteration][idx]
        auction.simulate_opportunity(publisher.name, curr_user_context, sigmoids[publisher.name], iteration, idx)

        mask[idx] = 1


def simulate_auctions_sequentially(
        publisher_list: List[Publisher], user_contexts: dict, sigmoids: dict, auction: Auction, i: int, rounds_per_iter: int
):
    # Simulate auctions sequentially
    for publisher in publisher_list:
        for j in range(rounds_per_iter):
            current_user_context = user_contexts[publisher.name][i][j]
            auction.simulate_opportunity(publisher.name, current_user_context, sigmoids[publisher.name], i, j)


def simulation_run(
        run, init_publisher_list, init_publisher_embeddings, user_contexts, sigmoids, auction, num_iter,
        rounds_per_iter, soglia_ctr, embedding_size, alpha, window_size
):
    start_time_run = time.time()
    agent_stats = pd.DataFrame()
    comb_linucb = SWCLinUCB(alpha=alpha, d=embedding_size, publisher_list=init_publisher_list, window_size=window_size)
    for i in range(num_iter):
        print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}')

        start_time = time.time()
        if i > 1:
            publisher_list = comb_linucb.round_iteration(
                curr_publisher_list=publisher_list,
                run=run,
                iteration=i,
                soglia_ctr=soglia_ctr
            )
        else:
            publisher_list = init_publisher_list
            comb_linucb.initial_round(run=run, iteration=i)
        print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}: Round iteration took {time.time() - start_time} seconds')

        start_time = time.time()
        simulate_auctions_sequentially(
            publisher_list=publisher_list,
            user_contexts=user_contexts,
            sigmoids=sigmoids,
            auction=auction,
            i=i,
            rounds_per_iter=rounds_per_iter
        )
        print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}: Simulate auctions took {time.time() - start_time} seconds')

        for agent_id, agent in enumerate(auction.agents):
            start_time = time.time()
            agent.update(iteration=i)
            # print(f'Run {run}, Iteration {i}: Agent update took {time.time() - start_time} seconds')

            if agent.name.startswith('Nostro'):
                start_time = time.time()
                agent_stats_pub = agent.iteration_stats_per_publisher()
                for publisher_data in agent_stats_pub:
                    comb_linucb.update(
                        publisher_name=publisher_data['publisher'],
                        publisher_embedding=init_publisher_embeddings[publisher_data['publisher']],
                        clicks=publisher_data['clicks'],
                        impressions=publisher_data['impressions'],
                        iteration=i
                    )
                print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}: Combinatorial LinUCB update took {time.time() - start_time} seconds')

                start_time = time.time()
                agent_df = pd.DataFrame(agent_stats_pub)
                agent_df['Agent'] = agent.name
                agent_df['Iteration'] = i
                agent_df['Run'] = run
                if i == 0:
                    agent_stats = agent_df
                else:
                    agent_stats = pd.concat([agent_stats, agent_df])
                print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}: Agent stats update took {time.time() - start_time} seconds')

            agent.clear_utility()
            agent.clear_logs()

        auction.clear_revenue()

    linucb_params = comb_linucb.linucb_params
    merged_df = pd.merge(
        agent_stats,
        linucb_params,
        on=['publisher', 'Iteration', 'Run'],
        how='left'
    )

    print(f'Run {run} took {time.time() - start_time_run} seconds')

    return agent_stats, merged_df


def run_simulation(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, adv_embeddings, alpha, window_size):
    print(f'[RUN {run}] Running simulation with soglia_ctr = {soglia_ctr} and alpha = {alpha}')

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    agent_stats, merged_df = simulation_run(run, init_publisher_list, init_publisher_embeddings, user_contexts, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, alpha, window_size)

    merged_df.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}_ws_{window_size}.csv'), index=False)


if __name__ == "__main__":
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rng.shuffle(publishers)
    init_publisher_list = publishers[:300]

    window_size_list = [50]
    alpha_list = [1]
    soglia_ctr = 0.97
    tasks = []
    for window_size in window_size_list:
        for alpha in alpha_list:
            for run in range(num_runs):
                tasks.append((output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, alpha, window_size))

    start_time = time.time()
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')

    # Save grouped results
    grouped_results = read_results(output_dir)
    grouped_results.to_csv(os.path.join(output_dir, 'grouped_results.csv'), index=False)
