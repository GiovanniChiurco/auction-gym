import multiprocessing
from CombinatorialLinUCB_nuo import CombinatorialLinUCBNuo
from CombinatorialLinUCB_giusto import CombinatorialLinUCBRight
from SWCLinUCB import SWCombinatorialLinUCBOpt
from new_main import *
import time
import pickle
import re

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

    return (rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size,
            embedding_var, obs_embedding_size, adv_embeddings, publisher_embeddings, rescaled_publisher_embeddings, window_size_list)

def simulate_auctions_sequentially(
        publisher_list: List[Publisher], sigmoids: dict, auction: Auction, i: int, rounds_per_iter: int
):
    # Simulate auctions sequentially
    for publisher in publisher_list:
        for j in range(rounds_per_iter):
            auction.simulate_opportunity(publisher.name, sigmoids[publisher.name], i, j)


def simulation_run(
        run, init_publisher_list, init_publisher_embeddings, sigmoids, auction, num_iter,
        rounds_per_iter, soglia_ctr, embedding_size, alpha, window_size
):
    start_time_run = time.time()
    agent_stats = pd.DataFrame()
    comb_linucb = SWCombinatorialLinUCBOpt(
        alpha=alpha, d=embedding_size, publisher_list=init_publisher_list, window_size=window_size
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
            comb_linucb.curr_superarm[i] = publisher_list
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

    linucb_theta_click, linucb_theta_impressions = comb_linucb.save_params()

    print(f'Run {run} took {time.time() - start_time_run} seconds')

    return agent_stats, merged_df, linucb_theta_click, linucb_theta_impressions


def run_simulation(output_dir, run, init_publisher_list, publisher_embeddings, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, obs_embedding_size, adv_embeddings, alpha, window_size):
    print(f'[RUN {run}] Running simulation with soglia_ctr = {soglia_ctr} and alpha = {alpha} and window_size = {window_size}')

    init_publisher_embeddings = {publisher.name: publisher_embeddings[publisher.name] for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    rescaled_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}

    agent_stats, merged_df, linucb_theta_click, linucb_theta_impressions = simulation_run(run, init_publisher_list, rescaled_publisher_embeddings, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, alpha, window_size)

    merged_df.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}_ws_{window_size}.csv'), index=False)
    
    # with open(os.path.join(output_dir, f'linucb_theta_click_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}_ws_{window_size}.pkl'), 'wb') as f:
    #     pickle.dump(linucb_theta_click, f)
    # with open(os.path.join(output_dir, f'linucb_theta_impressions_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}_ws_{window_size}.pkl'), 'wb') as f:
    #     pickle.dump(linucb_theta_impressions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    (rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var,
     obs_embedding_size, adv_embeddings, publisher_embeddings, rescaled_publisher_embeddings, window_size_list) = parse_config(args.config)
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values,
                                                                         agents, max_slots, embedding_size,
                                                                         embedding_var, obs_embedding_size)
    publishers = instantiate_publishers(rescaled_publisher_embeddings, rounds_per_iter)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter adv_embeddings
    adv_embeddings = {agent.adv_name: adv_embeddings[agent.adv_name] for agent in agents}

    rng.shuffle(publishers)
    
    num_pub = 300
    init_publisher_list = publishers[:num_pub]
    # Exclude the following publishers such that we always have publishers with at least 1 impression
    pub_to_exclude = ['dolcipassioni.net', 'healthy.thewom.it', 'unita.it', 'disboard.org', 'ilclubdellericette.it', 
                      'agrodolce.it', 'hovogliadidolce.it', 'giallozafferano.it', 'recetasgratis.net', 'prodottitipicitoscani.it', 
                      'wiadomosci.onet.pl', 'approdocalabria.it', 'buttalapasta.it']
    init_publisher_list = [pub for pub in init_publisher_list if pub.name not in pub_to_exclude]

    soglia_ctr_list = [0.9]
    alpha_list = [1]
    
    tasks = []
    for soglia_ctr in soglia_ctr_list:
        for window_size in window_size_list:
            for alpha in alpha_list:
                for run in range(num_runs):
                    tasks.append((output_dir, run, init_publisher_list, publisher_embeddings, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, obs_embedding_size, adv_embeddings, alpha, window_size))

    start_time = time.time()
    with multiprocessing.Pool(processes=6) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
