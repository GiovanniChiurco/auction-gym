import pandas as pd
from ortools.linear_solver import pywraplp
import numpy as np


def get_data(
        df: pd.DataFrame,
):
    # Estraggo i dati dal dataframe
    n = df.shape[0]
    clicks = df['clicks'].values
    rew_ucb = df['rew_ucb'].values
    impressions = df['impressions'].values
    spent = df['spent'].values
    cpc = df['cpc'].values
    return n, clicks, rew_ucb, impressions, spent, cpc

def solver(
        df: pd.DataFrame,
        n: int,
        rew_ucb: np.ndarray,
        clicks: np.ndarray = None,
        impressions: np.ndarray = None,
        spent: np.ndarray = None,
        cpc: np.ndarray = None,
        soglia_spent: float = None,
        soglia_clicks: float = None,
        soglia_cpc: float = None,
        soglia_ctr: float = None,
        soglia_num_publisher: int = None,
):
    # Inizializzo il solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    # Definisco le variabili
    x = [solver.BoolVar(f'x{i}') for i in range(n)]
    x_np = np.array(x)
    # Funzione obiettivo
    solver.Maximize(np.dot(rew_ucb, x_np))
    if soglia_clicks is not None:
        # Vincolo Clicks
        solver.Add(np.dot(clicks, x_np) >= soglia_clicks)
    if soglia_spent is not None:
        # Vincolo Spesa
        solver.Add(np.dot(spent, x_np) <= soglia_spent)
    if soglia_cpc is not None:
        # Vincolo CPC
        solver.Add(np.dot(cpc, x_np) <= soglia_cpc)
    if soglia_ctr is not None:
        # Vincolo CTR
        solver.Add(np.dot(rew_ucb, x_np) >= soglia_ctr * np.dot(impressions, x_np))
    if soglia_num_publisher is not None:
        # Vincolo Numero Publisher
        solver.Add(sum(x_np) <= soglia_num_publisher)
    # Risolvo il problema
    status = solver.Solve()
    results = pd.DataFrame(columns=df.columns)
    # Output dei risultati
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(n):
            if x[i].solution_value() == 1:
                if results.empty:
                    results = df.iloc[[i]]
                else:
                    results = pd.concat([results, df.iloc[[i]]])
        print("Knapsack Solver: Soluzione ottimale trovata!")
        print(f"Valore obiettivo = {round(solver.Objective().Value(), 3)}")
        print(f"CTR = {round(results['clicks'].sum() / results['impressions'].sum() * 100, 3)}%")
    else:
        print("Knapsack Solver: Non è stata trovata una soluzione ottimale.")
        # Return empty dataframe
        return pd.DataFrame()
    return results
