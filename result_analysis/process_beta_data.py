# Reads in data from txt file into a pandas dataframe and processes it
from intervaltree import IntervalTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich_dataframe import prettify
import os
import networkx as nx
import math

#|%%--%%| <Oew5vciEXq|zve2lMjqZv>

result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/QAOA_dat.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/qaoa.csv'

if os.path.exists(df_filename):
    qaoa_df = pd.read_csv(df_filename)
else:
    qaoa_df = pd.read_csv(result_filename)
    # qaoa_df['graph_num'] = qaoa_df['path'].str.extract(r'graph_(\d+)')
    # qaoa_df['graph_num'] = qaoa_df['graph_num'].astype(int)
    # qaoa_df['case'] = qaoa_df['random_path'].str.extract(r'(\w+).gml')
    qaoa_df.to_csv(df_filename, index=False)


def read_dfs(df_filename, result_filename):
    if os.path.exists(df_filename):
        df = pd.read_csv(df_filename)
    else:
        df = pd.read_csv(result_filename)
        df['graph_num'] = df['path'].str.extract(r'graph_(\d+)')
        df['graph_num'] = df['graph_num'].astype(int)
        df['C'] = qaoa_df['C']
        df.sort_values(by='graph_num', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(df_filename, index=False)
    return df


# Normal QAOA
result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/random_circuit/qaoa/out.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa.csv'
my_qaoa_df = read_dfs(df_filename, result_filename)

# Normal ma QAOA
result_filename_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/random_circuit/qaoa/out_ma.csv'
df_filename_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_ma.csv'
my_qaoa_ma_df = read_dfs(df_filename_ma, result_filename_ma)
# prettify(my_qaoa_ma_df, col_limit=15, row_limit=25)

# ma QAOA with rounded starting angles
result_filename_ma_post_rounded_4 = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_out_rounded_4.csv'
df_filename_ma_post_rounded_4 = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_ma_post_rounded_4.csv'
post_rounded_4_my_qaoa_ma_df = read_dfs(df_filename_ma_post_rounded_4, result_filename_ma_post_rounded_4)
# prettify(post_rounded_4_my_qaoa_ma_df, col_limit=15, row_limit=25)

# ma QAOA with random init starting angles
result_filename_random_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_out_random.csv'
df_filename_random_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_random.csv'
random_my_qaoa_ma_df = read_dfs(df_filename_random_ma, result_filename_random_ma)
# prettify(random_my_qaoa_ma_df, col_limit=15, row_limit=25)

# ma QAOA with random init starting angles and no optimization
result_filename_random_ma_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_out_random_no_opt.csv'
df_filename_random_ma_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_random_no_opt.csv'
no_opt_random_my_qaoa_ma_df = read_dfs(df_filename_random_ma_no_opt, result_filename_random_ma_no_opt)
# prettify(no_opt_random_my_qaoa_ma_df, col_limit=15, row_limit=25)

# ma QAOA with rounded starting angles and no optimization
result_filename_rounded_ma_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_out_rounded_4_no_opt.csv'
df_filename_rounded_ma_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_rounded_4_no_opt.csv'
no_opt_rounded_4_my_qaoa_ma_df = read_dfs(df_filename_rounded_ma_no_opt, result_filename_rounded_ma_no_opt)
# prettify(no_opt_rounded_4_my_qaoa_ma_df, col_limit=15, row_limit=25)

# ma QAOA with random rounded starting angles and no optimization
result_filename_random_rounded_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_out_random_rounded_no_opt.csv'
df_filename_random_rounded_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_random_rounded_no_opt.csv'
my_qaoa_ma_random_rounded_no_opt_df = read_dfs(df_filename_random_rounded_no_opt, result_filename_random_rounded_no_opt)

# 4 vertex graphs
ma_qaoa_4_vertex_result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_ma_4_vertex.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex.csv'
ma_qaoa_4_vertex_df = read_dfs(df_filename, ma_qaoa_4_vertex_result_filename)
prettify(ma_qaoa_4_vertex_df, col_limit=15, row_limit=25)

# 4 vertex graphs with rounded starting angles
ma_qaoa_4_vertex_rounded_result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_ma_4_vertex_rounded.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex_rounded.csv'
ma_qaoa_4_vertex_rounded_df = read_dfs(df_filename, ma_qaoa_4_vertex_rounded_result_filename)
prettify(ma_qaoa_4_vertex_rounded_df, col_limit=15, row_limit=25)

# 4 vertex graphs with random starting angles
ma_qaoa_4_vertex_random_result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_ma_4_vertex_random_int.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex_random.csv'
ma_qaoa_4_vertex_random_df = read_dfs(df_filename, ma_qaoa_4_vertex_random_result_filename)
prettify(ma_qaoa_4_vertex_random_df, col_limit=15, row_limit=25)

# 4 vertex graphs with random starting angles and no optimization
ma_qaoa_4_vertex_random_no_opt_result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_ma_4_vertex_random_int_no_opt.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex_random_no_opt.csv'
ma_qaoa_4_vertex_random_no_opt_df = read_dfs(df_filename, ma_qaoa_4_vertex_random_no_opt_result_filename)
prettify(ma_qaoa_4_vertex_random_no_opt_df)

# 4 vertex graphs with rounded starting angles and no optimization
ma_qaoa_4_vertex_rounded_no_opt_result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_ma_4_vertex_rounded_no_opt.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex_rounded_no_opt.csv'
ma_qaoa_4_vertex_rounded_no_opt_df = read_dfs(df_filename, ma_qaoa_4_vertex_rounded_no_opt_result_filename)
prettify(ma_qaoa_4_vertex_rounded_no_opt_df, col_limit=15, row_limit=25)

#|%%--%%| <zve2lMjqZv|dC3Dgw3N5c>

print('Mean AR:')
print(f'Normal ma-QAOA: {round(my_qaoa_ma_df["p_1"].mean(), 3)}')
# print(f'Ma-QAOA with rounded angles: {round(post_rounded_4_my_qaoa_ma_df["p_1"].mean(), 3)}')
print(f'Ma-QAOA with random init angles: {round(random_my_qaoa_ma_df["p_1"].mean(), 3)}')
print(f'Ma-QAOA with random init angles and no optimization: {round(no_opt_random_my_qaoa_ma_df["p_1"].mean(), 3)}')
# print(f'Ma-QAOA with rounded angles and no optimization: {round(no_opt_rounded_4_my_qaoa_ma_df["p_1"].mean(), 3)}')

print()
print(f'Normal ma-QAOA 4 vertex: {round(ma_qaoa_4_vertex_df["p_1"].mean(), 3)}')
# print(f'Ma-QAOA with rounded angles 4 vertex: {round(ma_qaoa_4_vertex_rounded_df["p_1"].mean(), 3)}')
print(f'Ma-QAOA with random init angles 4 vertex: {round(ma_qaoa_4_vertex_random_df["p_1"].mean(), 3)}')
print(f'Ma-QAOA with random init angles and no optimization 4 vertex: {round(ma_qaoa_4_vertex_random_no_opt_df["p_1"].mean(), 3)}')
# print(f'Ma-QAOA with rounded angles and no optimization 4 vertex: {round(ma_qaoa_4_vertex_rounded_no_opt_df["p_1"].mean(), 3)}')

print()

print('Standard deviation AR:')
print(f'Normal ma-QAOA: {round(my_qaoa_ma_df["p_1"].std(), 3)}')
# print(f'Ma-QAOA with rounded angles: {round(post_rounded_4_my_qaoa_ma_df["p_1"].std(), 3)}')
print(f'Ma-QAOA with random init angles: {round(random_my_qaoa_ma_df["p_1"].std(), 3)}')
print(f'Ma-QAOA with random init angles and no optimization: {round(no_opt_random_my_qaoa_ma_df["p_1"].std(), 3)}')
# print(f'Ma-QAOA with rounded angles and no optimization: {round(no_opt_rounded_4_my_qaoa_ma_df["p_1"].std(), 3)}')
print()
print(f'Normal ma-QAOA 4 vertex: {round(ma_qaoa_4_vertex_df["p_1"].std(), 3)}')
# print(f'Ma-QAOA with rounded angles 4 vertex: {round(ma_qaoa_4_vertex_rounded_df["p_1"].std(), 3)}')
print(f'Ma-QAOA with random init angles 4 vertex: {round(ma_qaoa_4_vertex_random_df["p_1"].std(), 3)}')
print(f'Ma-QAOA with random init angles and no optimization 4 vertex: {round(ma_qaoa_4_vertex_random_no_opt_df["p_1"].std(), 3)}')


#|%%--%%| <dC3Dgw3N5c|XkVFzAv3DG>

graphs = [
    b"CF",
    b"CU",
    b"CV",
    b"C]",
    b"C^",
    b"C~",
]

for i, g in enumerate(graphs):
    # G = nx.from_sparse6_bytes(g)
    G = nx.from_graph6_bytes(g)
    nx.write_gml(G, f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/graphs/main/all_4/graph_{i}/{i}.gml')
    print(G.edges())


def txt_to_df(filename, n=4):
    """Reads in txt file as a pandas dataframe"""
    m = int(n*(n-1)/2)
    columns = ['graph_num', 'maxcut_value', 'C_0', 'C_1', 'prob', 'p'] + [f'beta_{i}' for i in range(n)] + [f'gamma_{i}' for i in range(m)]
    df = pd.read_csv(filename, sep=',', header=None, names=columns)
    return df


n4_df = txt_to_df('result_analysis/QAOA_dat_n4.txt', n=4)
n8_df = txt_to_df('result_analysis/QAOA_dat.txt', n=8)

prettify(n4_df, col_limit=20)

#|%%--%%| <XkVFzAv3DG|9Nvdy9D2Ec>

n4_rounded_df = n4_df
# n4_rounded_df = round_angles(n4_df)
n4_angles = 0.25 * np.round(n4_rounded_df.filter(regex=('beta|gamma')).values.flatten() / 0.25)
# n4_angles = n4_rounded_df.filter(regex=('beta|gamma')).values.flatten()
print(n4_angles)
# plt.hist(n4_angles/np.pi)

n4_angles_dic = {}
for a in n4_angles:
    if np.isnan(a):
        continue
    elif a in n4_angles_dic.keys():
        n4_angles_dic[a] += 1
    else:
        n4_angles_dic[a] = 1

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

# x = [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]
# for a in x:
#     if a not in n4_angles_dic.keys():
#         n4_angles_dic[a] = 0
x = list(n4_angles_dic.keys())
y = list(np.array(list(n4_angles_dic.values()))/sum(n4_angles_dic.values())*100)

fig = plt.figure()
plt.bar(x, y, width=0.1, align='center')
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(x, rotation=45)
plt.xlabel('$k$')
plt.ylabel('Percentage of angles (%)')
plt.title('Percentage of angles rounded optimized at\n$k \pi$ for all four-vertex graphs', color=COLOR)
plt.savefig(f'{paper_dir}four_vertex_angles_new_new.eps', format='eps')
fig.show()


#|%%--%%| <9Nvdy9D2Ec|hsBT1qBLbE>

n8_df = txt_to_df('result_analysis/QAOA_dat.txt', n=8)
n8_rounded_df = n8_df

n8_angles = 0.25 * np.round(n8_rounded_df.filter(regex=('beta|gamma')).values.flatten() / (0.25*np.pi))

n8_angles_dic = {}
for a in n8_angles:
    if np.isnan(a):
        continue
    elif a in n8_angles_dic.keys():
        n8_angles_dic[a] += 1
    else:
        n8_angles_dic[a] = 1

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

x = list(n8_angles_dic.keys())
y = list(np.array(list(n8_angles_dic.values()))/sum(n8_angles_dic.values())*100)

fig = plt.figure()
plt.bar(x, y, width=0.1, align='center')
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(x, rotation=85)
plt.xlabel('$k$')
plt.ylabel('Percentage of angles (%)')
plt.title('Percentage of angles rounded optimized at\n$k \pi$ for all eight-vertex graphs', color=COLOR)
plt.savefig(f'{paper_dir}eight_vertex_angles_new_new.eps', format='eps')
fig.show()


#|%%--%%| <hsBT1qBLbE|EPEj1Mpz2D>
r"""°°°
Angle Dataframes
°°°"""
#|%%--%%| <EPEj1Mpz2D|9hiaMhKuId>


def normalize_qaoa_angles(angles):
    """
    Normalizes angles to the [-pi; pi] range.
    :param angles: QAOA angles.
    :return: Normalized angles.
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def make_angle_df(df, name, n, ma=True):
    df_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/{name}.csv'
    if os.path.exists(df_filename):
        angles_df = pd.read_csv(df_filename)
    else:
        if ma:
            if n == 8:
                nb = 8
                ng = 28
            elif n == 4:
                nb = 4
                ng = 6
        else:
            nb = 1
            ng = 1

        angles_df = df[['graph_num', 'C', 'p_1']]
        angles_df['p_1_angles'] = df['p_1_angles'].apply(lambda x: x.replace('[', '').replace(']', '').split(' '))
        angles_df['p_1_angles'] = angles_df['p_1_angles'].apply(lambda x: [float(a) for a in x if a != ''])
        angles_df['p_1_angles'] = angles_df['p_1_angles'].apply(lambda x: x[-nb:] + x[:-nb])  # put gamma angles after beta angles
        angles_df['p_1_angles'] = angles_df['p_1_angles'].apply(lambda x: normalize_qaoa_angles(np.array(x)))
        angles_df = pd.concat([df[['graph_num', 'C', 'p_1']], angles_df['p_1_angles'].apply(pd.Series)], axis=1)
        angles_df.rename(columns={i: f'beta_{i}' for i in range(nb)} | {i: f'gamma_{i-nb}' for i in range(nb, nb+ng)}, inplace=True)

        angles_df.to_csv(df_filename, index=False)

    return angles_df

#|%%--%%| <9hiaMhKuId|svYxweGbSI>


def correct_angle_rounding(df, norm=True):
    def nearest_fourth(ang):
        angles = np.pi/4 * np.round(ang * 4/np.pi)
        if norm:
            norm_angles = []
            for a in angles:
                if a <= -2 * np.pi:
                    while a <= -2 * np.pi:
                        a += 2 * np.pi
                    norm_angles.append(a)

                elif a >= 2 * np.pi:
                    while a >= 2 * np.pi:
                        a -= 2 * np.pi
                    norm_angles.append(a)

                else:
                    norm_angles.append(a)

            return np.array(norm_angles)

        else:
            return angles

    new_df = pd.DataFrame()
    # new_df['graph_num'] = df['graph_num']
    new_df['p_1_angles'] = df['p_1_angles'].apply(lambda x: np.array2string(nearest_fourth(np.fromstring(x[1:-1], sep=' '))))

    return new_df


my_qaoa_rounded_angles_df = correct_angle_rounding(my_qaoa_df)
my_qaoa_rounded_angles_ma_df = correct_angle_rounding(my_qaoa_ma_df)


my_qaoa_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_rounded_angles.csv', index=False)
my_qaoa_rounded_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_rounded_angles_ma.csv', index=False)

no_opt_random_my_qaoa_ma_rounded_angles_df = correct_angle_rounding(no_opt_random_my_qaoa_ma_df)
no_opt_random_my_qaoa_ma_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_random_rounded_angles.csv', index=False)

ma_qaoa_4_vertex_rounded_angles_df = correct_angle_rounding(ma_qaoa_4_vertex_df)
ma_qaoa_4_vertex_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex_rounded_angles.csv', index=False)

#|%%--%%| <svYxweGbSI|KS2J5Nr9GW>

prettify(no_opt_random_my_qaoa_ma_rounded_angles_df, col_limit=15, row_limit=5)
prettify(no_opt_random_my_qaoa_ma_df['p_1_angles'], col_limit=15, row_limit=5)

#|%%--%%| <KS2J5Nr9GW|wziRKfi2ym>


def random_initial_angles(df):

    new_df = pd.DataFrame()
    new_df['p_1_angles'] = df['p_1_angles'].apply(lambda x: np.array2string(np.array([np.random.randint(-8, 8) * np.pi/4 for _ in range(len(np.fromstring(x[1:-1], sep=' ')))])))
    return new_df


# my_qaoa_random_angles_df = random_initial_angles(my_qaoa_df)
my_qaoa_random_angles_ma_df = random_initial_angles(my_qaoa_ma_df)
# my_qaoa_random_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_random_angles.csv', index=False)
my_qaoa_random_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_random_angles_ma.csv', index=False)
ma_qaoa_4_vertex_random_angles_df = random_initial_angles(ma_qaoa_4_vertex_df)
ma_qaoa_4_vertex_random_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex_random_angles.csv', index=False)


#|%%--%%| <wziRKfi2ym|T3SbQ5jEdG>

prettify(my_qaoa_rounded_angles_ma_df.iloc[0], col_limit=15, row_limit=25)
prettify(my_qaoa_random_angles_ma_df.iloc[0], col_limit=15, row_limit=25)

#|%%--%%| <T3SbQ5jEdG|XzpAf9PxdX>

my_qaoa_angles_ma_df = make_angle_df(my_qaoa_ma_df, name='my_qaoa_angles_ma', ma=True)
post_rounded_my_qaoa_angles_ma_df = make_angle_df(post_rounded_my_qaoa_ma_df, name='post_rounded_my_qaoa_angles_ma', ma=True)
post_rounded_4_my_qaoa_angles_ma_df = make_angle_df(post_rounded_4_my_qaoa_ma_df, name='post_rounded_4_my_qaoa_angles_ma', ma=True)

# prettify(my_qaoa_angles_ma_df, col_limit=15, row_limit=25)
# prettify(post_rounded_my_qaoa_angles_ma_df, col_limit=15, row_limit=25)

rounded_my_qaoa_angles_ma_df = round_angles(my_qaoa_angles_ma_df)
rounded_post_rounded_my_qaoa_angles_ma_df = round_angles(post_rounded_my_qaoa_angles_ma_df)
rounded_post_rounded_4_my_qaoa_angles_ma_df = round_angles(post_rounded_4_my_qaoa_angles_ma_df)

prettify(rounded_my_qaoa_angles_ma_df, col_limit=15, row_limit=25)
prettify(rounded_post_rounded_my_qaoa_angles_ma_df, col_limit=15, row_limit=25)
prettify(rounded_post_rounded_4_my_qaoa_angles_ma_df, col_limit=15, row_limit=25)

#|%%--%%| <XzpAf9PxdX|6KsdbJ1XNz>
r"""°°°
For each graph:
1. enumerate edges
2. get common neighbors (triangles) and filter out those with no common neighbors (no triangles)
3. find gamma angles that are zero
4. check if zero gamma is in a triangle
5. add to dictionary
°°°"""
#|%%--%%| <6KsdbJ1XNz|CatgdrvSn1>

graph_dir = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/graphs/main/all_8/'


def do_zero_angles_correspond_to_triangles(df):
    def find_triangles(graph):
        edges = graph.edges()
        triangles = {}
        for i, e in enumerate(edges):
            triangles[i] = sorted(nx.common_neighbors(graph, *e))
        return {i: triangles[i] for i in triangles.keys() if len(triangles[i]) > 0}

    triangles = df['graph_num'].apply(lambda x: find_triangles(nx.Graph(nx.read_gml(graph_dir+f'graph_{x}/{x}.gml'))))
    gammas = df[df.filter(regex=('gamma')).columns]

    def get_zero_gammas(row):
        zero_gammas = []

        for i, g in enumerate(row):
            if g == 0:
                zero_gammas.append(i)

        return zero_gammas

    zero_gammas = gammas.apply(lambda x: get_zero_gammas(x), axis=1)

    return df['graph_num'].apply(lambda x: triangles.loc[x].keys() & zero_gammas.loc[x])


my_qaoa_angles_ma_df['triangles'] = do_zero_angles_correspond_to_triangles(my_qaoa_angles_ma_df)
post_rounded_my_qaoa_angles_ma_df['triangles'] = do_zero_angles_correspond_to_triangles(post_rounded_my_qaoa_angles_ma_df)
post_rounded_4_my_qaoa_angles_ma_df['triangles'] = do_zero_angles_correspond_to_triangles(post_rounded_4_my_qaoa_angles_ma_df)

prettify(my_qaoa_angles_ma_df['triangles'], col_limit=15, row_limit=25)
prettify(post_rounded_my_qaoa_angles_ma_df['triangles'], col_limit=15, row_limit=25)
prettify(post_rounded_4_my_qaoa_angles_ma_df['triangles'], col_limit=15, row_limit=25)

#|%%--%%| <CatgdrvSn1|kZq1xlc4nR>


def get_zero_gammas(row):
    zero_gammas = []

    for i, g in enumerate(row):
        if g == 0:
            zero_gammas.append(i)

    return zero_gammas


my_qaoa_angles_ma_df['zeros'] = my_qaoa_angles_ma_df[my_qaoa_angles_ma_df.filter(regex=('gamma')).columns].apply(get_zero_gammas, axis=1)
post_rounded_my_qaoa_angles_ma_df['zeros'] = post_rounded_my_qaoa_angles_ma_df[post_rounded_my_qaoa_angles_ma_df.filter(regex=('gamma')).columns].apply(get_zero_gammas, axis=1)

#|%%--%%| <kZq1xlc4nR|4O26vQ5nfy>

plot_df = my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Number of Triangles for Zero Gamma Angles')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/num_triangles_zero_gamma_angles_ma.eps')

#|%%--%%| <4O26vQ5nfy|8QLzlup78E>

plot_df = post_rounded_my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Number of Triangles for Zero Gamma Angles Post Rounded')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/num_triangles_zero_gamma_angles_ma_post_rounded.eps')

#|%%--%%| <8QLzlup78E|0TJyEkG09b>

plot_df = my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x)) / my_qaoa_angles_ma_df['zeros'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Fraction of Triangles for Zero Gamma Angles')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/fraction_triangles_zero_gamma_angles_ma.eps')


#|%%--%%| <0TJyEkG09b|JOt3dc2gxT>

plot_df = post_rounded_my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x)) / post_rounded_my_qaoa_angles_ma_df['zeros'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Fraction of Triangles for Zero Gamma Angles Post Rounded')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/fraction_triangles_zero_gamma_angles_ma_post_rounded.eps')

#|%%--%%| <JOt3dc2gxT|jyhkB7rqif>

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

# diff_ar = np.round(my_qaoa_ma_df['p_1'] - post_rounded_4_my_qaoa_ma_df['p_1'], 3)
# diff_ar = np.round(my_qaoa_ma_df['p_1'] - random_my_qaoa_ma_df['p_1'], 3)
# diff_ar = np.round(my_qaoa_ma_df['p_1'] - no_opt_random_my_qaoa_ma_df['p_1'], 3)
diff_ar = np.round(my_qaoa_ma_df['p_1'] - no_opt_rounded_4_my_qaoa_ma_df['p_1'], 3)
# diff_ar = np.round(my_qaoa_ma_df['p_1'] - my_qaoa_random_rounded_no_opt_df['p_1'], 3)
# diff_ar = pd.DataFrame({'AR^{ma} - AR^{ma, random init, no opt}': np.round(my_qaoa_ma_df['p_1'] - no_opt_random_my_qaoa_ma_df['p_1'], 3), 'AR^{ma} - AR^{ma, rounded, no opt}': np.round(my_qaoa_ma_df['p_1'] - no_opt_rounded_4_my_qaoa_ma_df['p_1'], 3), 'AR^{ma} - AR^{ma, random init}': np.round(my_qaoa_ma_df['p_1'] - random_my_qaoa_ma_df['p_1'], 3), 'AR^{ma} - AR^{ma, rounded}': np.round(my_qaoa_ma_df['p_1'] - post_rounded_4_my_qaoa_ma_df['p_1'], 3), 'AR^{ma} - AR^{ma}': np.round(my_qaoa_ma_df['p_1'] - my_qaoa_ma_df['p_1'], 3)})
diff_ar.value_counts().sort_index().plot()
# diff_ar.value_counts().sort_index().plot()
# diff_ar.value_counts().sort_index().plot()
# diff_ar.value_counts().sort_index().plot()
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(rotation=45)
plt.yscale('log')
# plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;rounded}}$')
# plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;random\;init}}$')
# plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;random\;init,\;no\;opt}}$')
plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;rounded,\;no\;opt}}$')
# plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;random\;init,\;rounded,\;no\;opt}}$')
plt.ylabel('Number of occurrences')
# plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;rounded}}$ and"+"\nthe number of times difference occurs for all eight-vertex graphs.", color=COLOR)
# plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;random\;init}}$ and"+"\nthe number of times difference occurs for all eight-vertex graphs.", color=COLOR)
# plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;random\;init,\;no\;opt}}$ and"+"\nthe number of times difference occurs for all eight-vertex graphs.", color=COLOR)
plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;rounded,\;no\;opt}}$ and"+"\nthe number of times difference occurs for all eight-vertex graphs.", color=COLOR)
# plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;random\;init,\;rounded,\;no\;opt}}$ and"+"\nthe number of times difference occurs for all eight-vertex graphs.", color=COLOR)
# plt.savefig(f'{paper_dir}difference_ar_eight_vertex_4.eps', format='eps')
# plt.savefig(f'{paper_dir}difference_ar_eight_vertex_random.eps', format='eps')
# plt.savefig(f'{paper_dir}difference_ar_eight_vertex_random_no_opt.eps', format='eps')
plt.savefig(f'{paper_dir}difference_ar_eight_vertex_rounded_no_opt.eps', format='eps')
# plt.savefig(f'{paper_dir}difference_ar_eight_vertex_random_rounded_no_opt.eps', format='eps')
# fig.show()


#|%%--%%| <jyhkB7rqif|b7ipgoXgap>

ar_df = pd.concat([my_qaoa_ma_df['p_1'], post_rounded_4_my_qaoa_ma_df['p_1'], random_my_qaoa_ma_df['p_1'], no_opt_rounded_4_my_qaoa_ma_df['p_1'], no_opt_random_my_qaoa_ma_df['p_1']], axis=1)
ar_df.columns = ['AR^{ma}', 'AR^{ma, rounded}', 'AR^{ma, random init}', 'AR^{ma, rounded, no opt}', 'AR^{ma, random init, no opt}']
ar_df.round(3)
ar_df.plot()
# ar_df.round(2).plot.bar()
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(rotation=45)
# plt.yscale('log')
plt.xlabel('$AR^{ma} - AR^{ma, r}$')
plt.ylabel('Number of occurrences')
plt.title('$AR^{ma}$, $AR^{ma, r}$ and $AR^{ma, i}$ and \nthe number of times difference occurs for all eight-vertex graphs.', color=COLOR)
plt.savefig('compare_ar_eight_vertex.eps', format='eps')
# fig.show()


#|%%--%%| <b7ipgoXgap|oKWOhyG1UM>

ar_gamma_df = pd.concat([my_qaoa_ma_df['p_1'], gamma_removed_ma_df['p_1'], random_gamma_removed_ma_df['p_1'], no_opt_random_gamma_removed_ma_df['p_1'], post_rounded_4_gamma_removed_ma_df['p_1']], axis=1)
ar_gamma_df.columns = ['AR^{ma}', 'AR^{ma, \gamma}', 'AR^{ma, \gamma, random init}', 'AR^{ma, \gamma, random init, no opt}', 'AR^{ma, \gamma, rounded}']
print("Mean AR with max degree edge removed")
print(ar_gamma_df.mean())
print("Standard deviation AR with max degree edge removed")
print(ar_gamma_df.std())

#|%%--%%| <oKWOhyG1UM|0nffVDOncJ>

diff_c = np.round(my_qaoa_angles_ma_df['C'] * my_qaoa_angles_ma_df['p_1'] - post_rounded_my_qaoa_angles_ma_df['C'] * post_rounded_my_qaoa_angles_ma_df['p_1'], 2)
diff_c.value_counts().sort_index().plot()
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(rotation=45)
plt.yscale('log')
plt.xlabel('$\\langle C \\rangle^{ma} -\\langle C \\rangle^{ma,r}$')
plt.ylabel('Number of occurrences')
plt.title('Difference between  $\\langle C \\rangle^{ma}$ and $\\langle C \\rangle^{ma,r}$ and \nthe number of times difference occurs for all eight-vertex graphs.', color=COLOR)
plt.savefig('difference_expval_eight_vertex.eps', format='eps')
# fig.show()


#|%%--%%| <0nffVDOncJ|HJwotsrdof>

plot_df = my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x))
print(plot_df.sum() / 39003)

plot_df = post_rounded_my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x))
print(plot_df.sum() / 43077)

plot_df = post_rounded_4_my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x))
print(plot_df.sum() / 43077)

#|%%--%%| <HJwotsrdof|JwQOUxWafz>


# def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
#     def gcd(a, b):
#         while b:
#             a, b = b, a % b
#         return a
#
#     def _multiple_formatter(x, pos):
#         den = denominator
#         num = int(np.rint(den*x/number))
#         com = gcd(num, den)
#         (num, den) = (int(num/com), int(den/com))
#         if den == 1:
#             if num == 0:
#                 return r'$0$'
#             if num == 1:
#                 return r'$%s$' % latex
#             elif num == -1:
#                 return r'$-%s$' % latex
#             else:
#                 return r'$%s%s$' % (num, latex)
#         else:
#             if num == 1:
#                 return r'$\frac{%s}{%s}$' % (latex, den)
#             elif num == -1:
#                 return r'$\frac{-%s}{%s}$' % (latex, den)
#             else:
#                 return r'$\frac{%s%s}{%s}$' % (num, latex, den)
#     return _multiple_formatter
#
#
# class Multiple:
#     def __init__(self, denominator=2, number=np.pi, latex='\pi'):
#         self.denominator = denominator
#         self.number = number
#         self.latex = latex
#
#     def locator(self):
#         return plt.MultipleLocator(self.number / self.denominator)
#
#     def formatter(self):
#         return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
#

P = my_qaoa_angles_df.plot(x='beta_0', y='gamma_0', kind='scatter', title='My QAOA', c='p_1', colormap='cool')
P.set_facecolor('black')

P = gamma_removed_angles_df.plot(x='beta_0', y='gamma_0', kind='scatter', title='My QAOA', c='p_1', colormap='cool')
P.set_facecolor('black')

P = my_qaoa_angles_ma_df.plot(x='beta_0', y='gamma_0', kind='scatter', title='My QAOA', c='p_1', colormap='cool')
P.set_facecolor('black')

P = gamma_removed_angles_ma_df.plot(x='beta_0', y='gamma_0', kind='scatter', title='My QAOA', c='p_1', colormap='cool')
P.set_facecolor('black')


#|%%--%%| <JwQOUxWafz|tOr8RXw3I6>

def plot_corr_matrix(df, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, vmax=1, vmin=-1, square=True, annot=False, ax=ax, cmap='cool', center=0, cbar_kws={'location': 'bottom'})
    # ax.yaxis.set_tick_params(rotation=0)
    # ax.xaxis.set_tick_params(rotation=45, left=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # plt.savefig(f'{paper_path}corr_matrix_{title}.eps')
    plt.show()


plot_corr_matrix(my_qaoa_angles_df, 'my_qaoa')
plot_corr_matrix(gamma_removed_angles_df, 'gamma_removed')
plot_corr_matrix(my_qaoa_angles_ma_df, 'my_qaoa_ma')
plot_corr_matrix(gamma_removed_angles_ma_df, 'gamma_removed_ma')
#|%%--%%| <tOr8RXw3I6|EZZVTQawMl>

prettify(my_qaoa_angles_df.round(3).describe())
prettify(gamma_removed_angles_df.round(3).describe())
prettify(my_qaoa_angles_ma_df.round(3).describe())
prettify(gamma_removed_angles_ma_df.round(3).describe())

#|%%--%%| <EZZVTQawMl|MRi3u3mQ0b>

good_num = 9
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
G = nx.Graph(nx.read_gml('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/'+gamma_removed_ma_df['path'].iloc[good_num]))
G_random = nx.Graph(nx.read_gml('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/'+gamma_removed_ma_df['random_path'].iloc[good_num]))
print(G.edges())
print(G_random.edges())

G_combined = nx.compose(G, G_random)

for e in G_combined.edges:
    if e in G.edges and e in G_random.edges:
        G_combined[e[0]][e[1]]['color'] = 'g'
    elif e in G.edges:
        G_combined[e[0]][e[1]]['color'] = 'r'
    else:
        G_combined[e[0]][e[1]]['color'] = 'b'

nx.draw(G_combined, with_labels=True, font_weight='bold', edge_color=[G_combined[u][v]['color'] for u, v in G_combined.edges])

#|%%--%%| <MRi3u3mQ0b|TWvJ5dOLxF>


def make_angle_df(df, name, ma=True):
    # df_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/{name}.csv'
    # if os.path.exists(df_filename):
    #     angles_df = pd.read_csv(df_filename)
    # else:
    if ma:
        nb = 4
        ng = 6
    else:
        nb = 1
        ng = 1

    angles_df = df[['p_1']]
    angles_df['p_1_angles'] = df['p_1_angles'].apply(lambda x: x.replace('[', '').replace(']', '').split(' '))
    angles_df['p_1_angles'] = angles_df['p_1_angles'].apply(lambda x: [float(a) for a in x if a != ''])
    angles_df['p_1_angles'] = angles_df['p_1_angles'].apply(lambda x: x[-nb:] + x[:-nb])  # put gamma angles after beta angles
    # angles_df['p_1_angles'] = angles_df['p_1_angles'].apply(lambda x: normalize_qaoa_angles(np.array(x)))
    angles_df = pd.concat([df[['p_1']], angles_df['p_1_angles'].apply(pd.Series)], axis=1)
    angles_df.rename(columns={i: f'beta_{i}' for i in range(nb)} | {i: f'gamma_{i-nb}' for i in range(nb, nb+ng)}, inplace=True)

    # angles_df.to_csv(df_filename, index=False)

    return angles_df


#|%%--%%| <TWvJ5dOLxF|knYB461DpI>

ma_qaoa_4_vertex_result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/angle_rounding_gamma_ma/normal_ma_4_vertex.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/ma_qaoa_4_vertex.csv'
ma_qaoa_4_vertex_df = read_dfs(df_filename, ma_qaoa_4_vertex_result_filename)
# ma_qaoa_4_vertex_df = pd.read_csv(ma_qaoa_4_vertex_result_filename)
prettify(ma_qaoa_4_vertex_df)

# rounded_ma_qaoa_4_vertex_df = correct_angle_rounding(ma_qaoa_4_vertex_df, norm=False)
# prettify(rounded_ma_qaoa_4_vertex_df)
n4_angles_df = make_angle_df(ma_qaoa_4_vertex_df, 'n4_angles', 4, ma=True)
prettify(n4_angles_df)

# n4_angles = n4_angles_df.filter(regex=('beta|gamma')).values.flatten()/(np.pi/4)
n4_angles = 0.25 * np.round(n4_angles_df.filter(regex=('beta|gamma')).values.flatten() / (0.25*np.pi))


def normalize_angles_2pi(angles):
    norm_angles = []
    for a in angles:
        if a <= -2:
            while a <= -2:
                a += 2
            norm_angles.append(a)

        elif a >= 2:
            while a >= 2:
                a -= 2
            norm_angles.append(a)

        else:
            norm_angles.append(a)

    return norm_angles


n4_angles = normalize_angles_2pi(n4_angles)

n4_angles_dic = {}
for a in n4_angles:
    if np.isnan(a):
        continue
    elif a in n4_angles_dic.keys():
        n4_angles_dic[a] += 1
    else:
        n4_angles_dic[a] = 1

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

# x = [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]
# for a in x:
#     if a not in n4_angles_dic.keys():
#         n4_angles_dic[a] = 0
x = list(n4_angles_dic.keys())
y = list(np.array(list(n4_angles_dic.values()))/sum(n4_angles_dic.values())*100)

fig = plt.figure()
plt.bar(x, y, width=0.1, align='center')
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(x, rotation=45)
plt.xlabel('$k$')
plt.ylabel('Percentage of angles (%)')
plt.title('Percentage of angles rounded optimized at\n$k \pi$ for all four-vertex graphs', color=COLOR)
plt.savefig(f'{paper_dir}my_four_vertex_angles.eps', format='eps')
plt.savefig(f'{paper_dir}my_four_vertex_angles.png', format='png')
fig.show()

#|%%--%%| <knYB461DpI|Z6mRwD1j4V>


result_filename_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/results/random_circuit/qaoa/out_ma.csv'
df_filename_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/my_qaoa_ma.csv'
my_qaoa_ma_df = read_dfs(df_filename_ma, result_filename_ma)
# prettify(my_qaoa_ma_df, col_limit=15, row_limit=25)

# rounded_ma_qaoa_4_vertex_df = correct_angle_rounding(ma_qaoa_4_vertex_df, norm=False)
# prettify(rounded_ma_qaoa_4_vertex_df)
n8_angles_df = make_angle_df(my_qaoa_ma_df, 'n8_angles', ma=True)
# prettify(n8_angles_df)

# n8_angles = n8_angles_df.filter(regex=('beta|gamma')).values.flatten()/(np.pi/8)
n8_angles = 0.25 * np.round(n8_angles_df.filter(regex=('beta|gamma')).values.flatten() / (0.25*np.pi))

# normalize anlgles between -2pi and 2pi


def normalize_angles_2pi(angles):
    norm_angles = []
    for a in angles:
        if a <= -2:
            while a <= -2:
                a += 2
            norm_angles.append(a)

        elif a >= 2:
            while a >= 2:
                a -= 2
            norm_angles.append(a)

        else:
            norm_angles.append(a)

    return norm_angles


n8_angles = normalize_angles_2pi(n8_angles)

n8_angles_dic = {}
for a in n8_angles:
    if np.isnan(a):
        continue
    elif a in n8_angles_dic.keys():
        n8_angles_dic[a] += 1
    else:
        n8_angles_dic[a] = 1

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

# x = [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]
# for a in x:
#     if a not in n8_angles_dic.keys():
#         n8_angles_dic[a] = 0
x = list(n8_angles_dic.keys())
y = list(np.array(list(n8_angles_dic.values()))/sum(n8_angles_dic.values())*100)

fig = plt.figure()
plt.bar(x, y, width=0.1, align='center')
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(x, rotation=85)
plt.xlabel('$k$')
plt.ylabel('Percentage of angles (%)')
plt.title('Percentage of angles rounded optimized at\n$k \pi$ for all eight-vertex graphs', color=COLOR)
plt.savefig(f'{paper_dir}my_eight_vertex_angles.eps', format='eps')
plt.savefig(f'{paper_dir}my_eight_vertex_angles.png', format='png')
fig.show()


#|%%--%%| <Z6mRwD1j4V|3YQeRN3hJK>

my_rounded_angles_4_vertex_df = correct_angle_rounding(ma_qaoa_4_vertex_df)
prettify(my_rounded_angles_4_vertex_df)

#|%%--%%| <3YQeRN3hJK|rLwkCwl2YW>

# TODO: run calculations of random initialization and rounding with 8 vertex graphs
# TODO: normalize angles between -2pi and 2pi and rerun the calculations
# TODO: write the discussion in LaTeX

