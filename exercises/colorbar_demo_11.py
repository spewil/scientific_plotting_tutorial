import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-dark')

# fig = plt.figure(2)
# ax = fig.add_subplot(111)
# colormap_name = 'viridis'
# img_plot = ax.imshow(
#     np.random.random((100, 100)), vmin=0, vmax=1, cmap=colormap_name)

# fig.subplots_adjust(bottom=0, top=1, left=0, right=.9)
# cb_ax = fig.add_axes([0.87, 0.05, 0.05, 0.865])
# cbar = fig.colorbar(img_plot, cax=cb_ax)

# # set the colorbar ticks and tick labels
# cbar.set_ticks(np.linspace(0, 1, 3, endpoint=True))
# cbar.set_ticklabels(['low', 'medium', 'high'])
# fig.tight_layout()
# plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2)
for ax in axes.flat:
    im = ax.imshow(
        np.random.random((100, 100)), vmin=0, vmax=1, cmap='viridis')

fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()
