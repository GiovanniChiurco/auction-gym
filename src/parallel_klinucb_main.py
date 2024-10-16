import concurrent.futures
from new_main import *
import multiprocessing

from kLinUCB import KLinUCB


def create_4d_array(dict_input):
    # Ottieni la lista delle chiavi e il numero di chiavi
    keys = list(dict_input.keys())
    num_keys = len(keys)
    
    # Assumi che tutti gli array abbiano la stessa dimensione
    n, m, k = dict_input[keys[0]].shape
    
    # Crea un array 4D vuoto
    array_4d = np.zeros((num_keys, n, m, k), dtype=np.float32)
    
    # Crea il dizionario per mappare le chiavi agli indici
    key_to_index = {}
    
    # Popola l'array 4D e aggiorna il dizionario di mappatura
    for i, key in enumerate(keys):
        array_4d[i] = dict_input[key]
        key_to_index[key] = i
    
    return key_to_index, array_4d

def create_5d_array(outer_key_to_index, nested_dict):
    # Ottieni la lista delle chiavi esterne e il numero di chiavi esterne
    outer_keys = list(outer_key_to_index.keys())
    t = len(outer_keys)
    
    # Ottieni la lista delle chiavi interne dal primo dizionario interno e il numero di chiavi interne
    first_inner_dict = nested_dict[outer_keys[0]]
    inner_keys = list(first_inner_dict.keys())
    s = len(inner_keys)
    
    # Assumi che tutti gli array abbiano la stessa dimensione (n, m, 1)
    n, m = first_inner_dict[inner_keys[0]].shape
    
    # Crea un array 5D vuoto
    array_5d = np.zeros((t, s, n, m), dtype=np.float32)
    
    # Crea i due dizionari per mappare le chiavi esterne e interne agli indici
    inner_key_to_index = {}
    
    # Popola l'array 5D e aggiorna i dizionari di mappatura
    for outer_key, i in outer_key_to_index.items():
        inner_dict = nested_dict[outer_key]
        for j, inner_key in enumerate(inner_keys):
            inner_key_to_index[inner_key] = j
            array_5d[i, j] = inner_dict[inner_key]
    
    return inner_key_to_index, array_5d


def simulate_auctions_random(
        publisher_list: List[Publisher], publisher_mapping: dict, ad_mapping: dict, user_contexts: dict, sigmoids:dict, auction: Auction, iteration: int
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
        
        pub_index = publisher_mapping[publisher.name]
        curr_user_context = user_contexts[pub_index][iteration][idx]
        auction.simulate_opportunity(publisher.name, curr_user_context, sigmoids[pub_index], ad_mapping, iteration, idx)

        mask[idx] = 1


def simulate_auctions_sequentially(
        publisher_list: List[Publisher], user_contexts: dict, sigmoids: dict, auction: Auction, i: int, rounds_per_iter: int
):
    # Simulate auctions sequentially
    for publisher in publisher_list:
        for j in range(rounds_per_iter):
            current_user_context = user_contexts[publisher.name][i][j]
            auction.simulate_opportunity(publisher.name, current_user_context, sigmoids[publisher.name], i, j)


def simulation_run(alpha, init_publisher_list, publisher_mapping, ad_mapping, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, embedding_size=70, k=300):
    agent_stats = pd.DataFrame()
    comb_linucb = KLinUCB(alpha=alpha, d=embedding_size, publisher_list=init_publisher_list, k=k)
    for i in range(num_iter):
        print(f'Iteration {i}')
        if i > 1:
            publisher_list = comb_linucb.round_iteration(
                init_publisher_list,
                iteration=i
            )
        else:
            comb_linucb.initial_round(publisher_list=init_publisher_list, iteration=i)
            publisher_list = init_publisher_list
        # Simulate auctions randomly
        simulate_auctions_random(
            publisher_list=publisher_list,
            publisher_mapping=publisher_mapping, 
            ad_mapping=ad_mapping,
            user_contexts=user_contexts,
            sigmoids=sigmoids,
            auction=auction,
            iteration=i
        )
        # Update agents bidder models and combinatorial LinUCB
        for agent_id, agent in enumerate(auction.agents):
            # Update agent
            agent.update(iteration=i)
            # Update LinUCB
            if agent.name.startswith('Nostro'):
                agent_stats_pub = agent.iteration_stats_per_publisher()
                for publisher_data in agent_stats_pub:
                    comb_linucb.update(
                        publisher_name=publisher_data['publisher'],
                        publisher_embedding=publisher_embeddings[publisher_data['publisher']],
                        reward=publisher_data['clicks'],
                        iteration=i
                    )
                agent_df = pd.DataFrame(agent_stats_pub)
                agent_df['Agent'] = agent.name
                agent_df['Iteration'] = i
                if i == 0:
                    agent_stats = agent_df
                else:
                    agent_stats = pd.concat([agent_stats, agent_df])

            agent.clear_utility()
            agent.clear_logs()

        auction.clear_revenue()

    linucb_params = comb_linucb.linucb_params
    merged_df = pd.merge(
        agent_stats,
        linucb_params,
        on=['publisher', 'Iteration'],
        how='outer'
    )
    return agent_stats, merged_df


def run_simulation(alpha, output_dir, init_publisher_list, embedding_size, publisher_mapping, ad_mapping, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, k):
    print(f'Running simulation alpha={alpha} and max num of publishers={k}')
    budget_results = simulation_run(alpha, init_publisher_list, publisher_mapping, ad_mapping, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, embedding_size=embedding_size, k=k)
    agent_stats, lin_ucb_params = budget_results

    lin_ucb_params.to_csv(
        os.path.join(output_dir, f'agent_stats_alpha_{alpha}_k_{k}.csv'), index=False)

    lin_ucb_params_train = lin_ucb_params[lin_ucb_params['Iteration'] > 1]
    group_iter = lin_ucb_params_train.groupby('Iteration') \
        .agg({'clicks': 'sum', 'exp_rew': 'sum', 'spent': 'sum', 'true_clicks': 'sum'}) \
        .reset_index()
    group_iter['abs_err'] = np.abs(group_iter['clicks'] - group_iter['exp_rew'])
    group_iter['abs_perc_err'] = np.abs(group_iter['clicks'] - group_iter['exp_rew']) / group_iter['clicks']

    group_iter.to_csv(
        os.path.join(output_dir, f'grouped_results_per_iter_alpha_{alpha}_maxnpub_{k}.csv'),
        index=False)

    return k, agent_stats, lin_ucb_params


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
    init_publisher_list = publishers[:200]
    ks = [10, 20, 50, 100, 150, 200, 250, 300]
    alpha_list = [0, 0.4, 1.0, 1.6]

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    # Convert dicts into numpy arrays
    publisher_mapping, user_contexts = create_4d_array(user_contexts)
    ad_mapping, sigmoids = create_5d_array(publisher_mapping, sigmoids)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    k_res = {}

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.starmap(run_simulation,
                               [(1, output_dir, init_publisher_list, embedding_size, publisher_mapping, ad_mapping, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, k)
                                for k in ks])

        for result in results:
            k_res[result[0]] = result[1:]
