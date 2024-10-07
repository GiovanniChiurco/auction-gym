import concurrent.futures
from new_main import *
import multiprocessing

from kLinUCB import KLinUCB


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


def simulation_run(alpha, init_publisher_list, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, embedding_size=70, k=300):
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


def run_simulation(alpha, output_dir, init_publisher_list, embedding_size, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, k):
    print(f'Running simulation alpha={alpha} and max num of publishers={k}')
    budget_results = simulation_run(alpha, init_publisher_list, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, embedding_size=embedding_size, k=k)
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
    ks = [10, 20, 50, 100, 150, 200, 250, 300]
    alpha_list = [0, 0.4, 1.0, 1.6]

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    for k in ks:
        for alpha in alpha_list:
            run_simulation(alpha, output_dir, init_publisher_list, embedding_size, user_contexts, sigmoids, publisher_embeddings, auction, num_iter, rounds_per_iter, k)
