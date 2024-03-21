import matplotlib.pyplot as plt

#|%%--%%| <nsqRpF6Oj8|aIcE2tXuWr>

paper_dir = '/home/vilcius//Papers/angle_analysis_ma_qaoa/paper/'

x = [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]
y = [2.04, 0, 12.24, 8.16, 10.20, 8.16, 14.29, 0, 2.04]

fig = plt.figure()
plt.bar(x, y, width=0.1, align='center')
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(x, rotation=45)
plt.xlabel('Multiple of $\pi$')
plt.ylabel('Percentage of angles (%)')
plt.title('Percentage of angles rounded optimized at\n$k \pi$ for all four-vertex graphs', color=COLOR)
plt.savefig(f'{paper_dir}four_vertex_angles.eps', format='eps')
fig.show()

#|%%--%%| <aIcE2tXuWr|P1DwhQ0w9v>

x = [-1.75, -1.5, -1.25, -1, -.75, -.5, -.25, 0, .25, .5, .75, 1, 1.25, 1.5, 1.75]
y = [0.19, 1.12, 0.34, 8.17, 0.91, 10.87, 10.07, 21.73, 10.07, 10.87, 0.91, 8.17, 0.34, 1.12, 0.19]

fig = plt.figure()
plt.bar(x, y, width=0.1, align='center')
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.xticks(x, rotation=45)
plt.xlabel('Multiple of $\pi$')
plt.ylabel('Percentage of angles (%)')
plt.title('Percentage of angles rounded optimized at\n$k \pi$ for all eight-vertex graphs', color=COLOR)
plt.savefig(f'{paper_dir}eight_vertex_angles.eps', format='eps')
fig.show()

#|%%--%%| <P1DwhQ0w9v|jfkJ25GpKs>

x = [0, 0.042893, 0.04702, 0.073223, 0.09404, 0.141061, 0.146477, 0.149519, 0.18934, 0.193467, 0.21967, 0.292893, 0.396447, 0.5]
y = [9836, 28, 795, 1, 59, 3, 288, 3, 1, 5, 1, 4, 4, 2]

fig = plt.figure()
plt.plot(x, y)
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
plt.savefig(f'{paper_dir}difference_expval_eight_vertex.eps', format='eps')
fig.show()

