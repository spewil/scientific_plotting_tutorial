import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

# 100 Axes objects

x = np.linspace(0, 20, 1000)
y = np.sin(x)

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

# zoombox plot
