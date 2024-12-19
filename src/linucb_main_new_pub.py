import multiprocessing
from CombinatorialLinUCB_giusto import CombinatorialLinUCBRight
from new_main import *
import time


def simulate_auctions_sequentially(
        publisher_list: List[Publisher], sigmoids: dict, auction: Auction, i: int, rounds_per_iter: int
):
    # Simulate auctions sequentially
    for publisher in publisher_list:
        for j in range(rounds_per_iter):
            auction.simulate_opportunity(publisher.name, sigmoids[publisher.name], i, j)


def simulation_run(
        run, 
        init_publisher_list, all_publisher_embeddings, sigmoids,
        new_pub_list,
        auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, alpha
):
    start_time_run = time.time()
    agent_stats = pd.DataFrame()
    comb_linucb = CombinatorialLinUCBRight(
        alpha=alpha, d=embedding_size, publisher_list=init_publisher_list
    )
    for i in range(num_iter):
        print(f'Run {run}, Iteration {i}, soglia_ctr = {soglia_ctr}, alpha = {alpha}')

        start_time = time.time()
        if i > 1:
            # Alla 250esima iterazione aggiungo i nuovi publisher
            if i == 250:
                publisher_list += new_pub_list
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
                
                comb_linucb.update(agent_stats_pub, all_publisher_embeddings)
                
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


def run_simulation(output_dir, run, init_publisher_list, new_pub_list, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, obs_embedding_size, adv_embeddings, alpha):
    print(f'[RUN {run}] Running simulation with soglia_ctr = {soglia_ctr} and alpha = {alpha}')
    all_publishers = init_publisher_list + new_pub_list
    all_publisher_embeddings = {publisher.name: publisher.embedding for publisher in all_publishers}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              all_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    agent_stats, merged_df = simulation_run(run,
                                            init_publisher_list, all_publisher_embeddings, sigmoids,
                                            new_pub_list,
                                            auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, alpha)

    merged_df.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.csv'), index=False)


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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rng.shuffle(publishers)
    
    num_pub = 300
    init_publisher_list = publishers[:num_pub]
    
    # Publisher da introdurre dopo 250 iterazioni
    new_pub_list = publishers[num_pub:400]

    soglia_ctr = 0.97
    alpha_list = [1]
    
    tasks = []
    for alpha in alpha_list:
        for run in range(num_runs):
            tasks.append((output_dir, run, init_publisher_list, new_pub_list, auction, num_iter, rounds_per_iter, soglia_ctr, embedding_size, obs_embedding_size, adv_embeddings, alpha))

    start_time = time.time()
    with multiprocessing.Pool(processes=1) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
