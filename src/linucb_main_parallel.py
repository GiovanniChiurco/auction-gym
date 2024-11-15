import multiprocessing
from CombinatorialLinUCB_nuo import CombinatorialLinUCBNuo
from CombinatorialLinUCB_giusto import CombinatorialLinUCBRight
from new_main import *
import time
import pickle
import re


# def parse_config(path):
#     with open(path) as f:
#         config = json.load(f)
#
#     # Set up Random Number Generator
#     rng = np.random.default_rng(config['random_seed'])
#     np.random.seed(config['random_seed'])
#
#     # Number of runs
#     num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1
#
#     # Max. number of slots in every auction round
#     # Multi-slot is currently not fully supported.
#     max_slots = 1
#
#     # Technical parameters for distribution of latent embeddings
#     embedding_size = config['embedding_size']
#     embedding_var = config['embedding_var']
#     obs_embedding_size = config['obs_embedding_size']
#
#     # Expand agent-config if there are multiple copies
#     agent_configs = []
#     num_agents = 0
#     for agent_config in config['agents']:
#         if 'num_copies' in agent_config.keys():
#             for i in range(1, agent_config['num_copies'] + 1):
#                 agent_config_copy = deepcopy(agent_config)
#                 agent_config_copy['name'] += f' {num_agents + 1}'
#                 agent_configs.append(agent_config_copy)
#                 num_agents += 1
#         else:
#             agent_configs.append(agent_config)
#             num_agents += 1
#
#     adv_embedding_path = config['adv_embedding_path']
#     adv_embeddings = pickle.load(open(adv_embedding_path, 'rb'))
#     # Adv embeddings for agents
#     agents2items = {
#         agent_config['name']: adv_embeddings[agent_config['adv_name']]
#         for agent_config in agent_configs
#     }
#     # Adv values for agents equal to 1.0 for all agents and advs
#     agents2item_values = {
#         agent_config['name']: np.array([1.0] * agent_config['num_items'], dtype=np.float32)
#         for agent_config in agent_configs
#     }
#     publisher_embeddings_path = config['publisher_embedding_path']
#     publisher_embeddings = pickle.load(open(publisher_embeddings_path, 'rb'))
#
#     obfuscated_publisher_embeddings_path = config['obfuscated_publisher_embedding_path']
#     obfuscated_publisher_embeddings = pickle.load(open(obfuscated_publisher_embeddings_path, 'rb'))
#
#     return (rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size,
#             embedding_var, obs_embedding_size, adv_embeddings, publisher_embeddings, obfuscated_publisher_embeddings)


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
        publisher_list: List[Publisher], sigmoids: dict, auction: Auction, i: int, rounds_per_iter: int
):
    # Simulate auctions sequentially
    for publisher in publisher_list:
        for j in range(rounds_per_iter):
            auction.simulate_opportunity(publisher.name, sigmoids[publisher.name], i, j)


def simulation_run(
        run, init_publisher_list, init_publisher_embeddings, sigmoids, auction, num_iter,
        rounds_per_iter, soglia_ctr, embedding_size, alpha
):
    start_time_run = time.time()
    agent_stats = pd.DataFrame()
    # comb_linucb = CombinatorialLinUCBNuo(alpha=alpha, d=embedding_size, publisher_list=init_publisher_list)
    comb_linucb = CombinatorialLinUCBRight(
        alpha=alpha, d=embedding_size, publisher_list=init_publisher_list
    )
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
            sigmoids=sigmoids,
            auction=auction,
            i=i,
            rounds_per_iter=rounds_per_iter
        )
        print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}: Simulate auctions took {time.time() - start_time} seconds')

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i)

            if agent.name.startswith('Nostro'):
                start_time = time.time()
                agent_stats_pub = agent.iteration_stats_per_publisher()
                # for publisher_data in agent_stats_pub:
                #     comb_linucb.update(
                #         publisher_name=publisher_data['publisher'],
                #         publisher_embedding=init_publisher_embeddings[publisher_data['publisher']],
                #         clicks=publisher_data['clicks'],
                #         impressions=publisher_data['impressions'],
                #         iteration=i
                #     )
                comb_linucb.update(agent_stats_pub, init_publisher_embeddings)
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

    # linucb_theta_click, linucb_theta_impressions = comb_linucb.save_params()

    print(f'Run {run} took {time.time() - start_time_run} seconds')

    return agent_stats, merged_df# , linucb_theta_click, linucb_theta_impressions


def run_simulation(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, obs_embedding_size, adv_embeddings, alpha):
    print(f'[RUN {run}] Running simulation with soglia_ctr = {soglia_ctr} and alpha = {alpha}')

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    # init_publisher_obfuscated_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_obfuscated_list}

    # agent_stats, merged_df, linucb_theta_click, linucb_theta_impressions = simulation_run(run, init_publisher_list, init_publisher_embeddings, user_contexts, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, alpha)
    # agent_stats, merged_df = simulation_run(run, init_publisher_obfuscated_list, init_publisher_obfuscated_embeddings, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, obs_embedding_size, alpha)
    agent_stats, merged_df = simulation_run(run, init_publisher_list, init_publisher_embeddings, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, alpha)

    merged_df.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.csv'), index=False)
    
    # with open(os.path.join(output_dir, f'model_params/linucb_theta_click_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.pkl'), 'wb') as f:
    #     pickle.dump(linucb_theta_click, f)
    # with open(os.path.join(output_dir, f'model_params/linucb_theta_impressions_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.pkl'), 'wb') as f:
    #     pickle.dump(linucb_theta_impressions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    (rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var,
     obs_embedding_size, adv_embeddings, publisher_embeddings, obfuscated_publisher_embeddings) = parse_config(args.config)
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values,
                                                                         agents, max_slots, embedding_size,
                                                                         embedding_var, obs_embedding_size)
    publishers = instantiate_publishers(publisher_embeddings, rounds_per_iter)
    # obfuscated_publishers = instantiate_publishers(obfuscated_publisher_embeddings, rounds_per_iter)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if not os.path.exists(os.path.join(output_dir, 'detailed_results')):
    #     os.makedirs(os.path.join(output_dir, 'detailed_results'))
    # if not os.path.exists(os.path.join(output_dir, 'model_params')):
    #     os.makedirs(os.path.join(output_dir, 'model_params'))

    rng.shuffle(publishers)
    # rng.shuffle(obfuscated_publishers)
    num_pub = 300
    init_publisher_list = publishers[:num_pub]
    # init_publisher_list_names = [pub.name for pub in init_publisher_list]
    # init_publisher_obfuscated_list = [pub_obf for pub_obf in obfuscated_publishers if pub_obf.name in init_publisher_list_names]

    soglia_ctr = 0.97
    alpha_list = [1]
    
    tasks = []
    for alpha in alpha_list:
        for run in range(num_runs):
            tasks.append((output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, obs_embedding_size, adv_embeddings, alpha))

    start_time = time.time()
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')

    # Save grouped results
    # grouped_results = read_results(output_dir)
    # grouped_results.to_csv(os.path.join(output_dir, 'grouped_results.csv'), index=False)
