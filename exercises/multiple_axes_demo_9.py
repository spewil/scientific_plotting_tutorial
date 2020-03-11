import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.random.random(10) * 1000
z = np.random.random(10) * 1000

fig = plt.figure()
ax = fig.add_subplot(111)

plot_objects = []
plot_objects.append(ax.plot(x, y, '-', label='y')[0])
plot_objects.append(ax.plot(x, z, '-.', label='z')[0])
# add something else with different range
ax_w = ax.twinx()
w = np.random.random(10)
plot_objects.append(ax_w.plot(x, w, '-r', label='w')[0])

# added these three lines
print(plot_objects)
plot_labels = [po.get_label() for po in plot_objects]
ax.legend(
    plot_objects,
    plot_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.125),
    ncol=3,
    fancybox=True,
    shadow=True)

ax.grid()
ax.set_xlabel(r"whatever $x$ is", FontSize=14)
ax.set_ylabel(r"whatever $y$ and $z$ are", FontSize=14)
ax_w.set_ylabel(r"whatever $w$ is", FontSize=14)
ax_w.set_ylim(0, 1)
ax.set_ylim(0, 1000)
fig.tight_layout()
plt.show()
