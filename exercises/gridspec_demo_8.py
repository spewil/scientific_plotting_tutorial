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

plt.show()

# Exercise -- Make the plot drawn on the board!

# Exercise -- Can you do it in a loop?
