import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-dark')

fig = plt.figure(2)
ax = fig.add_subplot(111)
img_plot = ax.imshow(np.random.random((100, 100)), vmin=0, vmax=1)

fig.subplots_adjust(bottom=0, top=1, left=0, right=.9)

cb_ax = fig.add_axes([0.87, 0.1, 0.05, 0.865])
cbar = fig.colorbar(img_plot, cax=cb_ax)

# set the colorbar ticks and tick labels
cbar.set_ticks(np.linspace(0, 1, 3, endpoint=True))
cbar.set_ticklabels(['low', 'medium', 'high'])
fig.tight_layout()
plt.show()
