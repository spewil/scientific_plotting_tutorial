import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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


"""
Extras: sometimes you want to create a legend for a subset of objects that are plotted,
or you want to create a legend with objects that are not plotted. 
You can do this by creating a list of patches and calling ax.legend()

In this example, each scatter point has two attributes that represent 
different information: 

(1) shape of the scatter point: represents the neuron number 
(2) color of the scatter point: represents whether the activity is before or after the stimulus 

"""


fig, ax = plt.subplots()
ax.scatter(1, 1, marker='>', color='blue')
ax.scatter(2, 1, marker='>', color='red')
ax.scatter(2, 2, marker='v', color='blue')
ax.scatter(1, 2, marker='v', color='red')

legend_elements = [Line2D([0], [0], marker='>',
                          color='w', label='Scatter',
                          markerfacecolor='black',
                          markersize=10),
                  Line2D([0], [0], marker='v',
                          color='w', label='Scatter',
                          markerfacecolor='black',
                          markersize=10),
                  Line2D([0], [0], lw=4,
                          color='blue'),
                  Line2D([0], [0], lw=4,
                          color='red'),
                  ]

labels = ['Neuron 1', 'Neuron 2',
         'Before stimulus', 'After stimulus']


ax.legend(legend_elements, labels)

plt.show()