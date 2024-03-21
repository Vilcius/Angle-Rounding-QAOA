"""
Entry points for large scale parallel calculation functions.
Used for angle rounding heuristic
"""

import os
from functools import partial
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.graph_utils import generate_graphs, generate_random_edge_graphs, generate_random_subgraphs, generate_remove_triangle_graphs, generate_remove_random_edge_from_max_degree_vertex, remove_2_random_edges_from_max_degree_vertex_other, remove_all_edges_from_max_degree_vertex
from src.parallel import optimize_expectation_parallel, WorkerBaseQAOA, WorkerMA, WorkerRandomCircuit


def init_dataframe(data_path: str, worker: WorkerBaseQAOA, out_path: str, random_type=None):
    if isinstance(worker, WorkerRandomCircuit):
        # paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{random_type}/{os.fsdecode(j)}')
        #          for i in range(11117) for j in sorted(os.listdir(os.fsencode(f'{data_path}graph_{i}/{random_type}')))]
        paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{i}.gml')
                 for i in range(11117)]
        index = pd.MultiIndex.from_tuples(paths, names=["path", "random_path"])
        df = DataFrame(index=index)

    elif worker.initial_guess_from is None:
        paths = [f'{data_path}/{i}.gml' for i in range(11117)]
        df = DataFrame(paths).set_axis(['path'], axis=1).set_index('path')

    elif isinstance(worker, WorkerMA):
        df = pd.read_csv(f'{data_path}/output/qaoa/constant/0.2/out.csv', index_col=0)
        # df = df.filter(regex=r'p_\d+_angles').rename(columns=lambda name: f'{name[:-7]}_starting_angles')
        df = df.filter(regex=r'p_\d+_angles')

    else:
        raise Exception('No init for this worker')
    df.to_csv(out_path)


def round_angles(angle_df):
    def nearest_eighth(ang):
        return (((round(((ang)/np.pi)/0.25))/8)*(2*np.pi)) % (2*np.pi)

    return angle_df.applymap(nearest_eighth)


def run_graphs_parallel():
    nodes = 8

    num_workers = 25
    # convergence_threshold = 0.9995
    convergence_threshold = 1.1
    reader = partial(nx.read_gml, destringizer=int)
    p = 1

    # for p in ps:
    random_type = 'angle_rounding_gamma'
    out_col = f'p_{p}'

    ### with optimization
    ## rounded QAOA
    # out_path = 'results/angle_rounding_gamma/normal_out_rounded_4.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_rounded_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## random QAOA
    # out_path = 'results/angle_rounding_gamma/normal_out_random.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_random_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## rounded ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/normal_out_rounded_4.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_rounded_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ## random ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/normal_out_random.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_random_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ## rounded gamma QAOA
    # out_path = 'results/angle_rounding_gamma/gamma_out_rounded_4.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_rounded_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## random gamma QAOA
    # out_path = 'results/angle_rounding_gamma/gamma_out_random.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_random_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## random gamma ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/gamma_out_rounded_4.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_rounded_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ## random gamma ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/gamma_out_random.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_random_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ### withOUT optimization
    ## rounded QAOA
    # out_path = 'results/angle_rounding_gamma/normal_out_rounded_4_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_rounded_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## random QAOA
    # out_path = 'results/angle_rounding_gamma/normal_out_random_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_random_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## rounded ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/normal_out_rounded_4_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_rounded_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ## random ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/normal_out_random_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/my_qaoa_random_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ## rounded gamma QAOA
    # out_path = 'results/angle_rounding_gamma/gamma_out_rounded_4_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_rounded_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## random gamma QAOA
    # out_path = 'results/angle_rounding_gamma/gamma_out_random_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_random_angles.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='qaoa')

    ## random gamma ma-QAOA
    # out_path = 'results/angle_rounding_gamma_ma/gamma_out_rounded_4_no_opt.csv'
    # angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_rounded_angles_ma.csv')
    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')

    ## random gamma ma-QAOA
    out_path = 'results/angle_rounding_gamma_ma/gamma_out_random_no_opt.csv'
    angle_df = pd.read_csv('/home/agwilkie/papers/angle_rounding/code/result_analysis/gamma_removed_random_angles_ma.csv')
    worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, angle_df=angle_df, search_space='ma')



    # angle_df = pd.read_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_rounded_angles_ma.csv')

    # initial_guess_from = None if p == 1 else f'p_{p - 1}'
    # initial_guess_from = f'p_{p}'
    # transfer_from = None if p == 1 else f'p_{p - 1}'
    # transfer_p = None if p == 1 else p - 1


    data_path = f'graphs/main/all_{nodes}/'

    def rows_func(df): return np.ones((df.shape[0], 1), dtype=bool) if p == 1 else df[f'p_{p - 1}'] < convergence_threshold

    out_folder = path.split(out_path)[0]
    if not path.exists(out_folder):
        os.makedirs(path.split(out_path)[0])
    if not path.exists(out_path):
        init_dataframe(data_path, worker, out_path, random_type)

    optimize_expectation_parallel(out_path, rows_func, num_workers, worker)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # generate_graphs()
    # for g in range(11117):
    #     remove_max_degree_edge(g)
    #     print(f'g = {g}')
    run_graphs_parallel()


