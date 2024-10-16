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
        run, init_publisher_list, init_publisher_embeddings, user_contexts, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr
):
    agent_stats = pd.DataFrame()
    comb_linucb = CombinatorialLinUCB(alpha=1, d=70, publisher_list=init_publisher_list)
    for i in range(num_iter):
        print(f'Iteration {i}')
        if i > 1:
            publisher_list = comb_linucb.round_iteration(
                publisher_list=publisher_list,
                iteration=i,
                soglia_ctr=soglia_ctr
            )
        else:
            publisher_list = init_publisher_list
            comb_linucb.initial_round(publisher_list=publisher_list, iteration=i)
        # Simulate auctions randomly
        simulate_auctions_random(
            publisher_list=publisher_list,
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
                        publisher_embedding=init_publisher_embeddings[publisher_data['publisher']],
                        reward=publisher_data['clicks'],
                        iteration=i
                    )
                agent_df = pd.DataFrame(agent_stats_pub)
                agent_df['Agent'] = agent.name
                agent_df['Iteration'] = i
                agent_df['Run'] = run
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
                comb_linucb.save_stats(grouped_data)

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


def run_simulation(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr):
    print(f'[RUN {run}] Running simulation with soglia_ctr = {soglia_ctr}')

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    budget_results = simulation_run(run, init_publisher_list, init_publisher_embeddings, user_contexts, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr)
    agent_stats, lin_ucb_params = budget_results

    lin_ucb_params.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_{soglia_ctr}.csv'), index=False)

    group_iter = lin_ucb_params.groupby(['Run', 'Iteration']) \
        .agg({'impressions': 'sum', 'clicks': 'sum', 'exp_rew': 'sum', 'spent': 'sum', 'true_clicks': 'sum'}) \
        .reset_index()
    group_iter['ctr'] = group_iter['clicks'] / group_iter['impressions']
    group_iter['true_ctr'] = group_iter['true_clicks'] / group_iter['impressions']
    group_iter['abs_err'] = np.abs(group_iter['clicks'] - group_iter['exp_rew'])
    group_iter['abs_perc_err'] = np.abs(group_iter['clicks'] - group_iter['exp_rew']) / group_iter['clicks']

    group_iter.to_csv(
        os.path.join(output_dir, f'grouped_results_per_iter_{run}_{soglia_ctr}.csv'),
        index=False)


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

    # init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    # start_gen_deal = time.time()
    # user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
    #                                           init_publisher_embeddings, adv_embeddings)
    # print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    # soglia_ctr_list = [0.5, 0.6, 0.7]#0.8, 0.9, 0.95, 0.97, 0.99]
    # for soglia_ctr in soglia_ctr_list:
    #     run_simulation(output_dir, init_publisher_list, user_contexts, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr)
    
    soglia_ctr = 0.97
    tasks = [(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr)
             for run in range(num_runs)]

    start_time = time.time()
    with multiprocessing.Pool(processes=1) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')
