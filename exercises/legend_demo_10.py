import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use("seaborn-dark")

x = np.linspace(0, 20, 1000)
y1 = np.sin(x)
y2 = np.cos(x)

# axes legend
fig = plt.figure(1)
ax = fig.add_subplot()
ax.plot(x, y1, "-b", label="sine")
ax.plot(x, y2, "-.r", label="cosine")
ax.legend(
    loc="upper left",
    ncol=2,
    mode="expand",
    title="I'm a Legend",
    edgecolor="red",
    facecolor="yellow",
    shadow=True,
    fancybox=True,  # rounded corners
    markerfirst=False,
    numpoints=10,
    fontsize='xx-small',
    bbox_to_anchor=(0, 0.5, 0.5, 0.5))
ax.set_ylim(-1.5, 2.0)

##

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
ax1.plot(x, y1, "-b", label="sine")
ax1.plot(x, y2, "-.r", label="cosine")
ax2.plot(x, np.random.random(len(x)), "-b")
ax2.plot(x, np.random.random(len(x)), "-.r")

# When bbox_to_anchor and loc are used together, the loc argument will inform matplotlib which part of the bounding box of the legend should be placed at the arguments of bbox_to_anchor.

ax2.set_xlabel("The $X$ Axis [$x$'s]", fontsize=14)
ax1.grid()
ax2.grid()
fig.suptitle("Figure Title", x=0.5, y=.95, fontsize=24)
fig.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.875, 0.95))
fig.tight_layout(rect=[0, 0, 1, .9])
# fig.subplots_adjust(bottom=0)
plt.show()

# Exercise -- change this plot to show the data provided
# E.g. subplot with legend for cells in top, behavioral data in bottom
# Axes legend and Figure legend with custom markers
