import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 1000)
y = np.sin(x)

# 100 Axes objects

fig = plt.figure(1)

axes = []
for _ in range(100):
    ax = plt.Axes(
        fig=fig,
        rect=[
            np.random.uniform(high=0, low=1),
            np.random.uniform(high=0, low=1), 0.25, 0.25
        ])
    ax.plot(x, y, "-b", label="sine")
    axes.append(ax)
    fig.add_axes(ax)

plt.show()

# manual zoombox

# proper zoombox plot
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig = plt.figure(1)
ax = fig.add_subplot(111)  # create a new figure with a default 111 subplot
ax.plot(x, y)
ax_inset = zoomed_inset_axes(
    ax, zoom=2, loc='lower left')  # , bbox_to_anchor=(left, bottom, w, h))
ax_inset.plot(x[50:100], y[50:100])
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()

# Exercise -- zoom in on ephys data provided
