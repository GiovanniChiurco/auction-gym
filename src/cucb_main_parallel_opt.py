import multiprocessing
from CUCBNuo import CUCBNuo
from new_main import *
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
    # Valori medi sulle run per ogni iterazione
    grouped_results = grouped_results_run_iter.groupby(['alpha', 'Iteration']).mean().reset_index()
    # intervallo di confidenza click
    grouped_results['clicks_std'] = grouped_results_run_iter.groupby(['alpha', 'Iteration']).std().reset_index()['clicks']
    grouped_results['clicks_ci'] = 1.96 * grouped_results['clicks_std'] / grouped_results_run_iter.groupby(['alpha', 'Iteration']).count().reset_index()['clicks']
    # intervallo di confidenza true_clicks
    grouped_results['true_clicks_std'] = grouped_results_run_iter.groupby(['alpha', 'Iteration']).std().reset_index()['true_clicks']
    grouped_results['true_clicks_ci'] = 1.96 * grouped_results['true_clicks_std'] / grouped_results_run_iter.groupby(['alpha', 'Iteration']).count().reset_index()['true_clicks']
    # intervallo di confidenza impressions
    grouped_results['impressions_std'] = grouped_results_run_iter.groupby(['alpha', 'Iteration']).std().reset_index()['impressions']
    grouped_results['impressions_ci'] = 1.96 * grouped_results['impressions_std'] / grouped_results_run_iter.groupby(['alpha', 'Iteration']).count().reset_index()['impressions']
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
        run, init_publisher_list, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, alpha
):
    agent_stats = pd.DataFrame()
    cucb = CUCBNuo(publisher_list=init_publisher_list, alpha=alpha)
    for i in range(num_iter):
        print(f'Iteration {i}, run {run}, alpha {alpha}, soglia_ctr {soglia_ctr}')
        initial_iteration = 1
        if i > initial_iteration:
            publisher_list = cucb.round_iteration(
                curr_publisher_list=publisher_list,
                soglia_ctr=soglia_ctr,
                run=run,
                iteration=i
            )
        else:
            cucb.set_time_t(i+1)
            # If for the first 3 iterations the agent cannot win any auction of a certain publisher, it is removed
            # if i == initial_iteration:
            #     publisher_list = cucb.get_selected_publishers()
            # else:
            publisher_list = init_publisher_list
        # Simulate auctions sequentially (faster)
        simulate_auctions_sequentially(
            publisher_list=publisher_list,
            sigmoids=sigmoids,
            auction=auction,
            i=i,
            rounds_per_iter=rounds_per_iter
        )
        # Update agents bidder models and combinatorial LinUCB
        for agent_id, agent in enumerate(auction.agents):
            # Update agent
            agent.update(iteration=i)
            # Update CUCB
            if agent.name.startswith('Nostro'):
                agent_stats_pub = agent.iteration_stats_per_publisher()
                for publisher_data in agent_stats_pub:
                    cucb.update_arm(
                        publisher_name=publisher_data['publisher'],
                        clicks=publisher_data['clicks'],
                        impressions=publisher_data['impressions']
                    )
                agent_df = pd.DataFrame(agent_stats_pub)
                agent_df['Agent'] = agent.name
                agent_df['Iteration'] = i
                agent_df['Run'] = run
                if i == 0:
                    agent_stats = agent_df
                else:
                    agent_stats = pd.concat([agent_stats, agent_df])

            agent.clear_utility()
            agent.clear_logs()

        auction.clear_revenue()

    cucb_est_click, cucb_est_impressions = cucb.save_params()

    cucb_est_ucb = cucb.est_ucb
    merged_df = pd.merge(
        agent_stats,
        cucb_est_ucb,
        on=['publisher', 'Iteration', 'Run'],
        how='left'
    )
    return agent_stats, merged_df, cucb_est_click, cucb_est_impressions


def run_simulation(output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, alpha, embedding_size, adv_embeddings):
    print(f'[RUN {run}] Running simulation with soglia_ctr = {soglia_ctr} e alpha = {alpha}')

    init_publisher_embeddings = {publisher.name: publisher.embedding for publisher in init_publisher_list}
    start_gen_deal = time.time()
    user_contexts, sigmoids = initialize_deal(num_iter, rounds_per_iter, embedding_size, 0.01,
                                              init_publisher_embeddings, adv_embeddings)
    print(f'Generating deal took {time.time() - start_gen_deal} seconds')

    budget_results = simulation_run(run, init_publisher_list, sigmoids, auction, num_iter, rounds_per_iter, soglia_ctr, alpha)
    agent_stats, lin_ucb_params, cucb_est_click, cucb_est_impressions = budget_results

    lin_ucb_params.to_csv(
        os.path.join(output_dir, f'agent_stats_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.csv'), index=False)
    
    with open(os.path.join(output_dir, f'cucb_est_click_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.pkl'), 'wb') as f:
        pickle.dump(cucb_est_click, f)
    with open(os.path.join(output_dir, f'cucb_est_impressions_run_{run}_ctr_{soglia_ctr}_alpha_{alpha}.pkl'), 'wb') as f:
        pickle.dump(cucb_est_impressions, f)


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
    # Exclude the following publishers such that we always have publishers with at least 1 impression
    pub_to_exclude = ['dolcipassioni.net', 'healthy.thewom.it', 'unita.it', 'disboard.org', 'ilclubdellericette.it', 
                      'agrodolce.it', 'hovogliadidolce.it', 'giallozafferano.it', 'recetasgratis.net', 'prodottitipicitoscani.it', 
                      'wiadomosci.onet.pl','approdocalabria.it', 'buttalapasta.it'
                      ]
    init_publisher_list = [pub for pub in init_publisher_list if pub.name not in pub_to_exclude]

    alpha_list = [1]
    soglia_ctr_list = [0.97]

    tasks = []
    for alpha in alpha_list:
        for soglia_ctr in soglia_ctr_list:
            for run in range(num_runs):
                tasks.append((output_dir, run, init_publisher_list, auction, num_iter, rounds_per_iter, soglia_ctr, alpha, embedding_size, adv_embeddings))

    start_time = time.time()
    with multiprocessing.Pool(processes=2) as pool:
        pool.starmap(run_simulation, tasks)
    print(f'Total time: {time.time() - start_time}')

    # Save grouped results
    # grouped_results = read_results(output_dir)
    # grouped_results.to_csv(os.path.join(output_dir, 'grouped_results.csv'), index=False)
