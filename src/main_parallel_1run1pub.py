import multiprocessing
from new_main import *


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
        run, init_publisher_list, user_contexts, sigmoids, auction, num_iter, rounds_per_iter
):
    agent_stats = pd.DataFrame()
    publisher_list=init_publisher_list
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
                agent_df['Iteration'] = i
                if i == 0:
                    agent_stats = agent_df
                else:
                    agent_stats = pd.concat([agent_stats, agent_df])
                grouped_data = agent_df.groupby(by=['publisher']) \
                    .sum() \
                    .reset_index() \
                    [['publisher', 'impressions', 'lost_auctions', 'clicks', 'spent']]
                grouped_data['cpc'] = grouped_data.apply(
                    lambda row: row['spent'] / row['clicks'] if row['clicks'] > 0 else 0, axis=1)

            agent.clear_utility()
            agent.clear_logs()

        auction.clear_revenue()
    return agent_stats


def run_simulation(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter):
    pub_name = init_publisher_list[0].name

    print(f'Running simulation with publishers: {pub_name}')

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)

    agent_stats = simulation_run(run, init_publisher_list, user_contexts, sigmoids, auction, num_iter, rounds_per_iter)

    agent_stats.to_csv(
        os.path.join(output_dir, f'agent_stats_pub_{pub_name}.csv'), index=False)


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
    
    init_publisher_list = publishers

    single_publishers = [[publisher] for publisher in init_publisher_list]
    
    tasks = [(output_dir, 1, single_publisher, auction, num_iter, rounds_per_iter)
             for single_publisher in single_publishers]

    start_time = time.time()
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
