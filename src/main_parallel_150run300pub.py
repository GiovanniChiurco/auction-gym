import multiprocessing
from new_main import *
import re
import gc


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
        run, output_dir, init_publisher_list, user_contexts, sigmoids, auction, num_iter, rounds_per_iter
):
    publisher_list=init_publisher_list
    output_file = os.path.join(output_dir, f'agent_stats_run_{run}.csv')
    for i in range(num_iter):
        print(f'Iteration {i}')
        # Simulate auctions randomly
        simulate_auctions_sequentially(
            publisher_list=publisher_list,
            user_contexts=user_contexts,
            sigmoids=sigmoids,
            auction=auction,
            i=i,
            rounds_per_iter=rounds_per_iter
        )
        # Update agents bidder models and combinatorial LinUCB
        for agent_id, agent in enumerate(auction.agents):
            # Update agent
            agent.update(iteration=i)
            # Update LinUCB
            if agent.name.startswith('Nostro'):
                agent_stats_pub = agent.iteration_stats_per_publisher()
                agent_df = pd.DataFrame(agent_stats_pub)
                agent_df['Agent'] = agent.name
                agent_df['Run'] = run
                agent_df['Iteration'] = i
                if i == 0:
                    agent_df.to_csv(output_file, mode='a', header=True, index=False)
                else:
                    agent_df.to_csv(output_file, mode='a', header=False, index=False)

            agent.clear_utility()
            agent.clear_logs()

        auction.clear_revenue()
        gc.collect()


def run_simulation(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter):
    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)

    simulation_run(run, output_dir, init_publisher_list, user_contexts, sigmoids, auction, num_iter, rounds_per_iter)

def read_pubs():
    dir = 'results/FP_Truthful_Oracle_sigmoids_cucb_est_click_impr_alphatune/'
    with open(dir + '300pubs.json') as f:
        pubs = json.load(f)
    return pubs


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

    all_pubs = read_pubs()
    init_publisher_list = [publisher for publisher in publishers if publisher.name in all_pubs]
    
    tasks = []
    for run in range(num_runs):
        tasks.append((output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter))

    start_time = time.time()
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
