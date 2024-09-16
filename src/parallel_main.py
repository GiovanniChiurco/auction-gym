from main import *
import argparse
import multiprocessing
import os
from collections import defaultdict
import numpy as np
import time


def measure_per_agent2df(run2agent2measure, measure_name):
    df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
    for run, agent2measure in run2agent2measure.items():
        for agent, measures in agent2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Agent'].append(agent)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)


def measure_per_agent2df_one_run(agent2measure: defaultdict, measure_name, run):
    df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
    for agent in agent2measure:
        for iteration, measure in enumerate(agent2measure[agent]):
            df_rows['Run'].append(run)
            df_rows['Agent'].append(agent)
            df_rows['Iteration'].append(iteration)
            df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)


def run_iteration(auction, num_iter, run, output_dir, rounds_per_iter):
    agent2net_utility = defaultdict(list)
    agent2gross_utility = defaultdict(list)
    agent2spent = defaultdict(list)

    auction_revenue = []

    with open(f'{output_dir}/run_{run}.csv', 'a') as f:
        f.write('Run,Iteration,Agent,Net Utility,Gross Utility,Spending\n')

        for i in range(num_iter):
            simulated_auctions_time = time.time()

            for _ in range(rounds_per_iter):
                auction.simulate_opportunity()

            print(f'[Run {run} - Iteration {i}]: Simulated auctions in {time.time() - simulated_auctions_time} seconds')

            for agent in auction.agents:

                update_time = time.time()

                agent.update(iteration=i)

                print(f'[Run {run} - Iteration {i}]: Updated agent {agent.name} in {time.time() - update_time} seconds')

                agent2net_utility[agent.name].append(agent.net_utility)
                agent2gross_utility[agent.name].append(agent.gross_utility)
                agent2spent[agent.name].append(agent.spending)
                # Salvare i risultati su file
                f.write(f'{run},{i},{agent.name},{agent.net_utility},{agent.gross_utility},{agent.spending}\n')
                f.flush()

                agent.clear_utility()
                agent.clear_logs()

            auction_revenue.append(auction.revenue)
            auction.clear_revenue()

            print(f'[Run {run} - Iteration {i}]: Finished iteration {i} in {time.time() - simulated_auctions_time} seconds')

    return agent2net_utility, agent2gross_utility, agent2spent, auction_revenue


def run_experiment(run, rng, agent_configs, agents2item_values, agents2items, config, max_slots, embedding_size, embedding_var, obs_embedding_size):
    print(f'Running experiment {run}')

    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents,
                                                           max_slots, embedding_size, embedding_var, obs_embedding_size)

    curr_iter_time = time.time()

    agent2net_utility, agent2gross_utility, agent2spent, auction_revenue = run_iteration(auction, num_iter, run, output_dir, rounds_per_iter)

    print(f'Finished experiment {run} in {time.time() - curr_iter_time} seconds')

    # Salvare i risultati su file
    net_utility_df = measure_per_agent2df_one_run(
        agent2net_utility, 'Net Utility', run).sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{num_iter}_iters_{run}_run.csv', index=False)

    return {'run': run, 'agent2net_utility': agent2net_utility, 'agent2gross_utility': agent2gross_utility,
            'agent2spent': agent2spent, 'auction_revenue': auction_revenue}


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(args.config)

    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14
    # Declare variables to plotting purposes
    rounds_per_iter = config['rounds_per_iter']
    num_iter = config['num_iter']

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2spent = {}

    run2auction_revenue = {}

    output_dir = config['output_dir']
    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Parallelizzare le simulazioni
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(run_experiment, [(run, rng, agent_configs, agents2item_values, agents2items, config,
                                                 max_slots, embedding_size, embedding_var, obs_embedding_size) for run in
                                                range(num_runs)])

        for result in results:
            run = result['run']
            run2agent2net_utility[run] = result['agent2net_utility']
            run2agent2gross_utility[run] = result['agent2gross_utility']
            run2auction_revenue[run] = result['auction_revenue']
            run2agent2spent[run] = result['agent2spent']

    def measure_per_agent2df(run2agent2measure, measure_name):
        df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
        for run, agent2measure in run2agent2measure.items():
            for agent, measures in agent2measure.items():
                for iteration, measure in enumerate(measures):
                    df_rows['Run'].append(run)
                    df_rows['Agent'].append(agent)
                    df_rows['Iteration'].append(iteration)
                    df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_per_agent(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.lineplot(data=df, x="Iteration", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    net_utility_df = plot_measure_per_agent(run2agent2net_utility, 'Net Utility').sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    gross_utility_df = plot_measure_per_agent(run2agent2gross_utility, 'Gross Utility').sort_values(['Agent', 'Run', 'Iteration'])
    gross_utility_df.to_csv(f'{output_dir}/gross_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    def measure2df(run2measure, measure_name):
        df_rows = {'Run': [], 'Iteration': [], measure_name: []}
        for run, measures in run2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_overall(run2measure, measure_name):
        # Generate DataFrame for Seaborn
        if type(run2measure) != pd.DataFrame:
            df = measure2df(run2measure, measure_name)
        else:
            df = run2measure
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        sns.lineplot(data=df, x="Iteration", y=measure_name, ax=axes)
        min_measure = min(0.0, np.min(df[measure_name]))
        max_measure = max(0.0, np.max(df[measure_name]))
        plt.xlabel('Iteration', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        factor = 1.1 if min_measure < 0 else 0.9
        plt.ylim(min_measure * factor, max_measure * 1.1)
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    auction_revenue_df = plot_measure_overall(run2auction_revenue, 'Auction Revenue')

    net_utility_df_overall = net_utility_df.groupby(['Run', 'Iteration'])['Net Utility'].sum().reset_index().rename(columns={'Net Utility': 'Social Surplus'})
    plot_measure_overall(net_utility_df_overall, 'Social Surplus')

    gross_utility_df_overall = gross_utility_df.groupby(['Run', 'Iteration'])['Gross Utility'].sum().reset_index().rename(columns={'Gross Utility': 'Social Welfare'})
    plot_measure_overall(gross_utility_df_overall, 'Social Welfare')

    auction_revenue_df['Measure Name'] = 'Auction Revenue'
    net_utility_df_overall['Measure Name'] = 'Social Surplus'
    gross_utility_df_overall['Measure Name'] = 'Social Welfare'

    columns = ['Run', 'Iteration', 'Measure', 'Measure Name']
    auction_revenue_df.columns = columns
    net_utility_df_overall.columns = columns
    gross_utility_df_overall.columns = columns

    pd.concat((auction_revenue_df, net_utility_df_overall, gross_utility_df_overall)).to_csv(f'{output_dir}/results_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    num_iter = config['num_iter']
    net_utility_df = measure_per_agent2df(
        run2agent2net_utility,
        'Net Utility').sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{num_iter}_iters_{num_runs}_runs.csv', index=False)
