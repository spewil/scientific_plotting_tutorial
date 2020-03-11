import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('seaborn-dark')

G = gridspec.GridSpec(3, 3)

axes_1 = plt.subplot(G[0, :])
axes_1.set_xticks([]), axes_1.set_yticks([])
axes_1.text(0.5, 0.5, 'Axes 1', ha='center', va='center', size=24, alpha=.5)

axes_2 = plt.subplot(G[1, :-1])
axes_2.set_xticks([]), axes_2.set_yticks([])
axes_2.text(0.5, 0.5, 'Axes 2', ha='center', va='center', size=24, alpha=.5)

axes_3 = plt.subplot(G[1:, -1])
axes_3.set_xticks([]), axes_3.set_yticks([])
axes_3.text(0.5, 0.5, 'Axes 3', ha='center', va='center', size=24, alpha=.5)

axes_4 = plt.subplot(G[-1, 0])
axes_4.set_xticks([]), axes_4.set_yticks([])
axes_4.text(0.5, 0.5, 'Axes 4', ha='center', va='center', size=24, alpha=.5)

axes_5 = plt.subplot(G[-1, -2])
axes_5.set_xticks([]), axes_5.set_yticks([])
axes_5.text(0.5, 0.5, 'Axes 5', ha='center', va='center', size=24, alpha=.5)

# plt.show()

# in a loop

rows = 5
cols = 2
num_points = 100
fig = plt.figure(figsize=(10, 10))  # inches?
gs = gridspec.GridSpec(rows, cols, figure=fig)
x = range(num_points)
axes_list = []
for ax_idx in range(rows * cols):
    y = np.random.normal(size=num_points)
    if ax_idx % 2 == 0:
        ax = fig.add_subplot(gs[ax_idx])
        ax.set_ylabel('signal')
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([str(-1), str(0), str(1)])
    else:
        ax = fig.add_subplot(gs[ax_idx], sharey=axes_list[-1])
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.plot(x, y)
    ax.set_xlabel('time [s]')
    ax.set_xticks([1, 50, 100])
    ax.set_xticklabels([str(1), str(50), str(100)])
    axes_list.append(ax)

fig.suptitle('Figure Title', size=20, x=0.5, y=.95)
gs.tight_layout(fig, rect=[0, 0, 1, .9])
fig.savefig("test.svg")

plt.show()

# notes:
# https://stackoverflow.com/questions/22511550/gridspec-with-shared-axes-in-python
# https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
