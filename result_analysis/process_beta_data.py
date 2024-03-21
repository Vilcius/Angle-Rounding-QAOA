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

#|%%--%%| <Oew5vciEXq|XkVFzAv3DG>

graphs = [
    b"CF",
    b"CU",
    b"CV",
    b"C]",
    b"C^",
    b"C~",
]

for g in graphs:
    # G = nx.from_sparse6_bytes(g)
    G = nx.from_graph6_bytes(g)
    print(G.edges())


def txt_to_df(filename, n=4):
    """Reads in txt file as a pandas dataframe"""
    m = int(n*(n-1)/2)
    columns = ['graph_num', 'maxcut_value', 'C_0', 'C_1', 'prob', 'p'] + [f'beta_{i}' for i in range(n)] + [f'gamma_{i}' for i in range(m)]
    df = pd.read_csv(filename, sep=',', header=None, names=columns)
    return df


n4_df = txt_to_df('./QAOA_dat_n4.txt', n=4)
n8_df = txt_to_df('./QAOA_dat.txt', n=8)

prettify(n4_df, col_limit=20)

#|%%--%%| <XkVFzAv3DG|XUQMKqAi40>

beta_4_df = n4_df[['graph_num', 'maxcut_value', 'C_1'] + [f'beta_{i}' for i in range(4)]]
beta_8_df = n8_df[['graph_num', 'maxcut_value', 'C_1'] + [f'beta_{i}' for i in range(8)]]

prettify(beta_4_df.round(3), col_limit=15)

#|%%--%%| <XUQMKqAi40|qGPNptVx1f>

# group each beta value for each row if they are within 0.01 of each other
beta_groups = beta_8_df.groupby([beta_8_df[f'beta_{i}'].round(2) for i in range(8)])
beta_groups = beta_4_df.groupby([beta_4_df[f'beta_{i}'].round(2) for i in range(4)])

prettify(pd.DataFrame(beta_groups))

for name, group in beta_groups:
    print(name)
    print(group)
    print('\n\n')
#|%%--%%| <qGPNptVx1f|bIuGW5hjL6>


def group_betas(df, tolerance=0.01, signed=False):
    t = {g: IntervalTree() for g in range(df.shape[0])}
    for col in df.columns[3:]:
        for g in t:
            beta = df[col].iloc[g]
            if signed:
                beta = np.abs(beta)
            t[g][beta-tolerance:beta+tolerance] = col
    result = {}
    for g in t:
        result[g] = {}
        for col in df.columns[3:]:
            beta = df[col].iloc[g]
            if signed:
                beta = np.abs(beta)
            betas = [iv.data for iv in t[g][beta]]
            if betas not in result[g].values():
                result[g][np.round(beta, 3)] = betas
    return result


beta_group_4 = group_betas(beta_4_df, tolerance=0.01, signed=False)
beta_group_8 = group_betas(beta_8_df, signed=False)

# prettify(beta_group_8)
prettify(pd.DataFrame(beta_group_4).T, col_limit=20)

#|%%--%%| <bIuGW5hjL6|BLyXUOcrHe>

gamma_4_df = n4_df[['graph_num', 'maxcut_value', 'C_1'] + [f'gamma_{i}' for i in range(6)]]
gamma_8_df = n8_df[['graph_num', 'maxcut_value', 'C_1'] + [f'gamma_{i}' for i in range(28)]]

prettify(gamma_4_df.round(3), col_limit=15)


def group_gammas(df, tolerance=0.01, signed=False):
    t = {g: IntervalTree() for g in range(df.shape[0])}
    for col in df.columns[3:]:
        for g in t:
            gamma = df[col].iloc[g]
            if signed:
                gamma = np.abs(gamma)
            t[g][gamma-tolerance:gamma+tolerance] = col
    result = {}
    for g in t:
        result[g] = {}
        for col in df.columns[3:]:
            gamma = df[col].iloc[g]
            if signed:
                gamma = np.abs(gamma)
            gammas = [iv.data for iv in t[g][gamma]]
            print(gamma, gammas)
            if gammas not in result[g].values():
                result[g][np.round(gamma, 3)] = gammas
    return result


gamma_group_4 = group_gammas(gamma_4_df, tolerance=0.01, signed=False)
# gamma_group_8 = group_gammas(gamma_8_df, signed=False)


gamma_group_4 = pd.DataFrame(gamma_group_4).T
gamma_group_4 = gamma_group_4[[c for c in gamma_group_4.columns if not math.isnan(c)]]
prettify(gamma_group_4, col_limit=20)


#|%%--%%| <BLyXUOcrHe|dcmLQlFsw0>

prettify(gamma_8_df.round(3), col_limit=15)
gamma_group_8 = group_gammas(gamma_8_df, tolerance=0.5, signed=False)
gamma_group_8 = pd.DataFrame(gamma_group_8).T
prettify(gamma_group_8, col_limit=10)
gamma_group_8 = gamma_group_8[[c for c in gamma_group_8.columns if not math.isnan(c)]]
prettify(gamma_group_8.tail(10), col_limit=20)


#|%%--%%| <dcmLQlFsw0|TwxFLVaGvr>
r"""°°°
Analysis the gamma removed results
°°°"""
#|%%--%%| <TwxFLVaGvr|KwlEXhLWzx>

result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/QAOA_dat.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/qaoa.csv'

if os.path.exists(df_filename):
    qaoa_df = pd.read_csv(df_filename)
else:
    qaoa_df = pd.read_csv(result_filename)
    # qaoa_df['graph_num'] = qaoa_df['path'].str.extract(r'graph_(\d+)')
    # qaoa_df['graph_num'] = qaoa_df['graph_num'].astype(int)
    # qaoa_df['case'] = qaoa_df['random_path'].str.extract(r'(\w+).gml')
    qaoa_df.to_csv(df_filename, index=False)

prettify(qaoa_df, col_limit=15, row_limit=25)

#|%%--%%| <KwlEXhLWzx|nvwfR5NWiq>

result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/qaoa/out.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa.csv'
if os.path.exists(df_filename):
    my_qaoa_df = pd.read_csv(df_filename)
else:
    my_qaoa_df = pd.read_csv(result_filename)
    my_qaoa_df['graph_num'] = my_qaoa_df['path'].str.extract(r'graph_(\d+)')
    my_qaoa_df['graph_num'] = my_qaoa_df['graph_num'].astype(int)
    my_qaoa_df['C'] = qaoa_df['C']
    my_qaoa_df.sort_values(by='graph_num', ascending=True, inplace=True)
    my_qaoa_df.reset_index(drop=True, inplace=True)
    my_qaoa_df.to_csv(df_filename, index=False)

prettify(my_qaoa_df, col_limit=15, row_limit=25)


#|%%--%%| <nvwfR5NWiq|r9KGv6lzB9>


result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/qaoa/out_ma.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_ma.csv'
if os.path.exists(df_filename):
    my_qaoa_ma_df = pd.read_csv(df_filename)
else:
    my_qaoa_ma_df = pd.read_csv(result_filename)
    my_qaoa_ma_df['graph_num'] = my_qaoa_ma_df['path'].str.extract(r'graph_(\d+)')
    my_qaoa_ma_df['graph_num'] = my_qaoa_ma_df['graph_num'].astype(int)
    my_qaoa_ma_df['C'] = qaoa_df['C']
    my_qaoa_ma_df.sort_values(by='graph_num', ascending=True, inplace=True)
    my_qaoa_ma_df.reset_index(drop=True, inplace=True)
    my_qaoa_ma_df.to_csv(df_filename, index=False)

prettify(my_qaoa_ma_df, col_limit=15, row_limit=25)

#|%%--%%| <r9KGv6lzB9|o8woCAUB5o>


result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma/out.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed.csv'
if os.path.exists(df_filename):
    gamma_removed_df = pd.read_csv(df_filename)
else:
    gamma_removed_df = pd.read_csv(result_filename)
    gamma_removed_df['graph_num'] = gamma_removed_df['path'].str.extract(r'graph_(\d+)')
    gamma_removed_df['graph_num'] = gamma_removed_df['graph_num'].astype(int)
    gamma_removed_df['C'] = qaoa_df['C']
    gamma_removed_df.sort_values(by='graph_num', ascending=True, inplace=True)
    gamma_removed_df.reset_index(drop=True, inplace=True)
    gamma_removed_df.to_csv(df_filename, index=False)

prettify(gamma_removed_df, col_limit=15, row_limit=25)


#|%%--%%| <o8woCAUB5o|bDfGXf95XD>


result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/out.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_ma.csv'
if os.path.exists(df_filename):
    gamma_removed_ma_df = pd.read_csv(df_filename)
else:
    gamma_removed_ma_df = pd.read_csv(result_filename)
    gamma_removed_ma_df['graph_num'] = gamma_removed_ma_df['path'].str.extract(r'graph_(\d+)')
    gamma_removed_ma_df['graph_num'] = gamma_removed_ma_df['graph_num'].astype(int)
    gamma_removed_ma_df['C'] = qaoa_df['C']
    gamma_removed_ma_df.sort_values(by='graph_num', ascending=True, inplace=True)
    gamma_removed_ma_df.reset_index(drop=True, inplace=True)
    gamma_removed_ma_df.to_csv(df_filename, index=False)

prettify(gamma_removed_ma_df, col_limit=15, row_limit=25)


#|%%--%%| <bDfGXf95XD|q159S1GpbB>

graphs_better = (gamma_removed_df['p_1'] > (qaoa_df['C_1']/qaoa_df['C'])*qaoa_df['prob'])
graphs_better_ma = (gamma_removed_ma_df['p_1'] > (qaoa_df['C_1']/qaoa_df['C'])*qaoa_df['prob'])
my_graphs_better = (gamma_removed_df['p_1'] > my_qaoa_df['p_1'])
my_graphs_better_ma = (gamma_removed_ma_df['p_1'] > my_qaoa_ma_df['p_1'])
number_better = graphs_better.sum()
number_better_ma = graphs_better_ma.sum()
my_number_better = my_graphs_better.sum()
my_number_better_ma = my_graphs_better_ma.sum()
the_graphs_better = graphs_better[graphs_better].index
the_graphs_better_ma = graphs_better_ma[graphs_better_ma].index
my_the_graphs_better = my_graphs_better[my_graphs_better].index
my_the_graphs_better_ma = my_graphs_better_ma[my_graphs_better_ma].index

print(f'Number of better results: {number_better}')
# print(f'Graphs with better results: {list(the_graphs_better)}')
print(f'Number of better results ma: {number_better_ma}')
# print(f'Graphs with better results ma: {list(the_graphs_better_ma)}')

print(f'My number of better results: {my_number_better}')
print(f'My number of better results ma: {my_number_better_ma}')

# prettify(gamma_removed_df[graphs_better])
# prettify(qaoa_df[graphs_better])

good_num = 1

fig, ax = plt.subplots(int(np.ceil(my_number_better_ma/6)), 6, figsize=(20, 2*int(np.ceil(my_number_better_ma/2))))
for good_num in range(my_number_better_ma):
    G = nx.Graph(nx.read_gml('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/'+gamma_removed_ma_df['path'].iloc[my_the_graphs_better_ma[good_num]]))
    G_random = nx.Graph(nx.read_gml('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/'+gamma_removed_ma_df['random_path'].iloc[my_the_graphs_better_ma[good_num]]))
    # print(G.edges())
    # print(G_random.edges())

    G_combined = nx.compose(G, G_random)

    for e in G_combined.edges:
        if e in G.edges and e in G_random.edges:
            G_combined[e[0]][e[1]]['color'] = 'g'
        elif e in G.edges:
            G_combined[e[0]][e[1]]['color'] = 'r'
        else:
            G_combined[e[0]][e[1]]['color'] = 'b'

    nx.draw(G_combined, ax=ax[good_num//6, good_num % 6], with_labels=True, font_weight='bold', edge_color=[G_combined[u][v]['color'] for u, v in G_combined.edges])
plt.savefig(f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/good_graphs_ma.eps')


#|%%--%%| <q159S1GpbB|VELj7QMBXg>

print('max p_1:', gamma_removed_df['p_1'].max())
print('max p_1 ma:', gamma_removed_ma_df['p_1'].max())

print()
print('min p_1:', gamma_removed_df['p_1'].min())
print('min p_1 ma:', gamma_removed_ma_df['p_1'].min())

print()
print('mean p_1:', gamma_removed_df['p_1'].mean())
print('mean p_1 ma:', gamma_removed_ma_df['p_1'].mean())

print()
print('number better gamma vs gamma_ma:', (gamma_removed_df['p_1'] < gamma_removed_ma_df['p_1']).sum())
print('percentage better gamma vs gamma_ma:', (gamma_removed_df['p_1'] < gamma_removed_ma_df['p_1']).mean())

#|%%--%%| <VELj7QMBXg|EPEj1Mpz2D>
r"""°°°
Angle Dataframes
°°°"""
#|%%--%%| <EPEj1Mpz2D|W8rt1w8UnD>

# Load Angle DataFrames
gamma_removed_angles_df = pd.read_csv('./result_analysis/gamma_removed_angles.csv')
gamma_removed_angles_ma_df = pd.read_csv('./result_analysis/gamma_removed_angles_ma.csv')
my_qaoa_angles_df = pd.read_csv('./result_analysis/my_qaoa_angles.csv')
my_qaoa_angles_ma_df = pd.read_csv('./result_analysis/my_qaoa_angles_ma.csv')


#|%%--%%| <W8rt1w8UnD|9hiaMhKuId>


def normalize_qaoa_angles(angles):
    """
    Normalizes angles to the [-pi; pi] range.
    :param angles: QAOA angles.
    :return: Normalized angles.
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def make_angle_df(df, name, ma=True):
    df_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/{name}.csv'
    if os.path.exists(df_filename):
        angles_df = pd.read_csv(df_filename)
    else:
        if ma:
            nb = 8
            ng = 28
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


gamma_removed_angles_df = make_angle_df(gamma_removed_df, name='gamma_removed_angles', ma=False)
gamma_removed_angles_ma_df = make_angle_df(gamma_removed_ma_df, name='gamma_removed_angles_ma', ma=True)
my_qaoa_angles_df = make_angle_df(my_qaoa_df, name='my_qaoa_angles', ma=False)
my_qaoa_angles_ma_df = make_angle_df(my_qaoa_ma_df, name='my_qaoa_angles_ma', ma=True)

prettify(gamma_removed_angles_ma_df, col_limit=15, row_limit=25)
prettify(gamma_removed_angles_df, col_limit=15, row_limit=25)
prettify(my_qaoa_angles_df, col_limit=15, row_limit=25)
prettify(my_qaoa_angles_ma_df, col_limit=15, row_limit=25)

#|%%--%%| <9hiaMhKuId|I5SC4wjez3>

gamma_removed_angles_df = pd.read_csv('./result_analysis/gamma_removed_angles.csv')
gamma_removed_angles_ma_df = pd.read_csv('./result_analysis/gamma_removed_angles_ma.csv')
my_qaoa_angles_df = pd.read_csv('./result_analysis/my_qaoa_angles.csv')
my_qaoa_angles_ma_df = pd.read_csv('./result_analysis/my_qaoa_angles_ma.csv')


def round_angles(angle_df):
    def nearest_eighth(ang):
        # ang = eval(ang)
        angles = np.pi/8 * np.round(ang * 8/np.pi)
        return np.arctan2(np.sin(angles), np.cos(angles))

    def nearest_fourth(ang):
        # ang = eval(ang)
        angles = np.pi/4 * np.round(ang * 4/np.pi)
        return np.arctan2(np.sin(angles), np.cos(angles))

    # return angle_df.filter(regex=('(beta_\d|gamma_\d)')).apply(lambda x: nearest_eighth(x))
    # angle_df[angle_df.filter(regex=('beta_|gamma_')).columns] = angle_df.filter(regex=('beta_|gamma_')).apply(lambda x: nearest_eighth(x))
    angle_df[angle_df.filter(regex=('beta_|gamma_')).columns] = angle_df.filter(regex=('beta_|gamma_')).apply(lambda x: nearest_fourth(x))
    return angle_df


def angles_to_array_string(angle_df):
    angle_df['p_1_angles'] = angle_df.filter(regex=('beta_\d|gamma_\d')).values.tolist()  # .apply(lambda x: np.array2string(x))
    angle_df['p_1_angles'] = angle_df['p_1_angles'].apply(lambda x: [*filter(pd.notna, x)])
    angle_df['p_1_angles'] = angle_df['p_1_angles'].apply(lambda x: np.array(x))
    angle_df['p_1_angles'] = angle_df['p_1_angles'].apply(lambda x: np.array2string(x))
    return angle_df


prettify(my_qaoa_angles_ma_df)
prettify(round_angles(my_qaoa_angles_ma_df))
prettify(angles_to_array_string(round_angles(my_qaoa_angles_ma_df))['p_1_angles'])

my_qaoa_rounded_angles_df = angles_to_array_string(round_angles(my_qaoa_angles_df))
my_qaoa_rounded_angles_ma_df = angles_to_array_string(round_angles(my_qaoa_angles_ma_df))
gamma_removed_rounded_angles_df = angles_to_array_string(round_angles(gamma_removed_angles_df))
gamma_removed_rounded_angles_ma_df = angles_to_array_string(round_angles(gamma_removed_angles_ma_df))

my_qaoa_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_rounded_angles.csv', index=False)
my_qaoa_rounded_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_rounded_angles_ma.csv', index=False)
gamma_removed_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_rounded_angles.csv', index=False)
gamma_removed_rounded_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_rounded_angles_ma.csv', index=False)


#|%%--%%| <I5SC4wjez3|ny8rRe9WAK>

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


result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/qaoa/out.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa.csv'
my_qaoa_df = read_dfs(df_filename, result_filename)

result_filename_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/qaoa/out_ma.csv'
df_filename_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_ma.csv'
my_qaoa_ma_df = read_dfs(df_filename_ma, result_filename_ma)
prettify(my_qaoa_ma_df, col_limit=15, row_limit=25)

result_filename_ma_post_rounded_4 = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/normal_out_rounded_4.csv'
df_filename_ma_post_rounded_4 = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_ma_post_rounded_4.csv'
post_rounded_4_my_qaoa_ma_df = read_dfs(df_filename_ma_post_rounded_4, result_filename_ma_post_rounded_4)
prettify(post_rounded_4_my_qaoa_ma_df, col_limit=15, row_limit=25)

result_filename_random_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/normal_out_random.csv'
df_filename_random_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_random.csv'
random_my_qaoa_ma_df = read_dfs(df_filename_random_ma, result_filename_random_ma)
prettify(random_my_qaoa_ma_df, col_limit=15, row_limit=25)

result_filename_random_ma_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/normal_out_random_no_opt.csv'
df_filename_random_ma_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_random_no_opt.csv'
no_opt_random_my_qaoa_ma_df = read_dfs(df_filename_random_ma_no_opt, result_filename_random_ma_no_opt)
prettify(no_opt_random_my_qaoa_ma_df, col_limit=15, row_limit=25)

result_filename_rounded_ma_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/normal_out_rounded_4_no_opt.csv'
df_filename_rounded_ma_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_rounded_4_no_opt.csv'
no_opt_rounded_4_my_qaoa_ma_df = read_dfs(df_filename_rounded_ma_no_opt, result_filename_rounded_ma_no_opt)
prettify(no_opt_rounded_4_my_qaoa_ma_df, col_limit=15, row_limit=25)

#|%%--%%| <ny8rRe9WAK|zLt2WbtQVi>

result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma/out.csv'
df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed.csv'
gamma_removed_df = read_dfs(df_filename, result_filename)

result_filename_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/out.csv'
df_filename_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_ma.csv'
gamma_removed_ma_df = read_dfs(df_filename_ma, result_filename_ma)

result_filename_ma_post_rounded_4 = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/gamma_out_rounded_4.csv'
df_filename_ma_post_rounded_4 = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_ma_post_rounded_4.csv'
post_rounded_4_gamma_removed_ma_df = read_dfs(df_filename_ma_post_rounded_4, result_filename_ma_post_rounded_4)

result_filename_random_ma = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/gamma_out_random.csv'
df_filename_random_ma = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_random.csv'
random_gamma_removed_ma_df = read_dfs(df_filename_random_ma, result_filename_random_ma)

result_filename_random_ma_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/gamma_out_random_no_opt.csv'
df_filename_random_ma_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_random_no_opt.csv'
no_opt_random_gamma_removed_ma_df = read_dfs(df_filename_random_ma_no_opt, result_filename_random_ma_no_opt)

result_filename_rounded_ma_no_opt = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/angle_rounding_gamma_ma/gamma_out_rounded_4_no_opt.csv'
df_filename_rounded_ma_no_opt = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_rounded_4_no_opt.csv'
no_opt_rounded_4_gamma_removed_ma_df = read_dfs(df_filename_rounded_ma_no_opt, result_filename_rounded_ma_no_opt)


#|%%--%%| <zLt2WbtQVi|svYxweGbSI>


def correct_angle_rounding(df):
    def nearest_fourth(ang):
        # ang = eval(ang)
        angles = np.pi/4 * np.round(ang * 4/np.pi)
        return np.arctan2(np.sin(angles), np.cos(angles))

    new_df = pd.DataFrame()
    # new_df['graph_num'] = df['graph_num']
    new_df['p_1_angles'] = df['p_1_angles'].apply(lambda x: np.array2string(nearest_fourth(np.fromstring(x[1:-1], sep=' '))))

    return new_df


my_qaoa_rounded_angles_df = correct_angle_rounding(my_qaoa_df)
my_qaoa_rounded_angles_ma_df = correct_angle_rounding(my_qaoa_ma_df)
gamma_removed_rounded_angles_df = correct_angle_rounding(gamma_removed_df)
gamma_removed_rounded_angles_ma_df = correct_angle_rounding(gamma_removed_ma_df)


my_qaoa_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_rounded_angles.csv', index=False)
my_qaoa_rounded_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_rounded_angles_ma.csv', index=False)
gamma_removed_rounded_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_rounded_angles.csv', index=False)
gamma_removed_rounded_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_rounded_angles_ma.csv', index=False)


#|%%--%%| <svYxweGbSI|wziRKfi2ym>

def random_initial_angles(df):

    new_df = pd.DataFrame()
    new_df['p_1_angles'] = df['p_1_angles'].apply(lambda x: np.array2string(np.array([np.random.randint(-4, 3) * np.pi/4 for _ in range(len(np.fromstring(x[1:-1], sep=' ')))])))
    return new_df


my_qaoa_random_angles_df = random_initial_angles(my_qaoa_df)
my_qaoa_random_angles_ma_df = random_initial_angles(my_qaoa_ma_df)
my_qaoa_random_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_random_angles.csv', index=False)
my_qaoa_random_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/my_qaoa_random_angles_ma.csv', index=False)

gamma_removed_random_angles_df = random_initial_angles(gamma_removed_df)
gamma_removed_random_angles_ma_df = random_initial_angles(gamma_removed_ma_df)
gamma_removed_random_angles_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_random_angles.csv', index=False)
gamma_removed_random_angles_ma_df.to_csv('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/gamma_removed_random_angles_ma.csv', index=False)

#|%%--%%| <wziRKfi2ym|T3SbQ5jEdG>

prettify(my_qaoa_rounded_angles_ma_df.iloc[0], col_limit=15, row_limit=25)
prettify(my_qaoa_random_angles_ma_df.iloc[0], col_limit=15, row_limit=25)

#|%%--%%| <T3SbQ5jEdG|zS78lMKq0p>


num_better = (post_rounded_my_qaoa_ma_df['p_1'].round(3) >= my_qaoa_ma_df['p_1'].round(3)).sum()
print(f'Number of better results: {num_better}')

#|%%--%%| <zS78lMKq0p|XzpAf9PxdX>

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

graph_dir = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/graphs/main/all_8/'


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
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/num_triangles_zero_gamma_angles_ma.eps')

#|%%--%%| <4O26vQ5nfy|8QLzlup78E>

plot_df = post_rounded_my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Number of Triangles for Zero Gamma Angles Post Rounded')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/num_triangles_zero_gamma_angles_ma_post_rounded.eps')

#|%%--%%| <8QLzlup78E|0TJyEkG09b>

plot_df = my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x)) / my_qaoa_angles_ma_df['zeros'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Fraction of Triangles for Zero Gamma Angles')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/fraction_triangles_zero_gamma_angles_ma.eps')


#|%%--%%| <0TJyEkG09b|JOt3dc2gxT>

plot_df = post_rounded_my_qaoa_angles_ma_df['triangles'].apply(lambda x: len(x)) / post_rounded_my_qaoa_angles_ma_df['zeros'].apply(lambda x: len(x))
plot_df.plot(kind='hist', title='Fraction of Triangles for Zero Gamma Angles Post Rounded')
plt.savefig('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/result_analysis/fraction_triangles_zero_gamma_angles_ma_post_rounded.eps')

#|%%--%%| <JOt3dc2gxT|jyhkB7rqif>

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

# diff_ar = np.round(my_qaoa_ma_df['p_1'] - post_rounded_my_qaoa_ma_df['p_1'], 10)
# diff_ar = np.round(my_qaoa_ma_df['p_1'] - post_rounded_4_my_qaoa_ma_df['p_1'], 3)
diff_ar = np.round(my_qaoa_ma_df['p_1'] - random_my_qaoa_ma_df['p_1'], 3)
# diff_ar = np.round(post_rounded_4_my_qaoa_ma_df['p_1'] - random_my_qaoa_ma_df['p_1'], 3)
# diff_ar = np.round(my_qaoa_ma_df['p_1'] - no_opt_random_my_qaoa_ma_df['p_1'], 3)
# diff_ar = np.round(my_qaoa_ma_df['p_1'] - no_opt_rounded_4_my_qaoa_ma_df['p_1'], 3)
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
# plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;rounded,\;no\;opt}}$')
plt.xlabel(r'$AR^{\mathrm{ma}} - AR^{\mathrm{ma,\;random\;init}}$')
plt.ylabel('Number of occurrences')
# plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;rounded,\;no\;opt}}$ and"+"the number of times difference occurs for all eight-vertex graphs.", color=COLOR)
plt.title(r"Difference between $AR^{\mathrm{ma}}$ and $AR^{\mathrm{ma,\;random\;init}}$ and"+"\nthe number of times difference occurs for all eight-vertex graphs.", color=COLOR)
# plt.savefig(f'{paper_dir}difference_ar_eight_vertex_4.eps', format='eps')
# plt.savefig('difference_ar_eight_vertex_4.eps', format='eps')
# plt.savefig(f'{paper_dir}difference_ar_eight_vertex_rounded_no_opt.eps', format='eps')
plt.savefig(f'{paper_dir}difference_ar_eight_vertex_random.eps', format='eps')
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
G = nx.Graph(nx.read_gml('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/'+gamma_removed_ma_df['path'].iloc[good_num]))
G_random = nx.Graph(nx.read_gml('/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/'+gamma_removed_ma_df['random_path'].iloc[good_num]))
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

