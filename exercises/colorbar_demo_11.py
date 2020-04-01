import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-dark')

# # auto cax
# fig, axes = plt.subplots(nrows=2, ncols=2)
# for ax in axes.flat:
#     im = ax.imshow(
#         np.random.random((100, 100)), vmin=0, vmax=1, cmap='gist_earth')
# fig.colorbar(im, ax=axes.ravel().tolist())
# plt.show()

# # custom cax
# data = np.arange(100, 0, -1).reshape(10, 10)
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.1, top=0.8, left=0, right=1)
# cax = fig.add_axes([0.24, .9, 0.5, 0.05])
# im = ax.imshow(data, cmap='gist_earth')
# fig.colorbar(im, cax=cax, orientation='horizontal')
# plt.show()
''' the lesson here is to use auto when possible? '''

fig = plt.figure(2)
ax = fig.add_subplot(111)
colormap_name = 'viridis'
img_plot = ax.imshow(
    np.random.random((100, 100)), vmin=0, vmax=1, cmap=colormap_name)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=.9)
cb_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
cbar = fig.colorbar(img_plot, cax=cb_ax)

# set the colorbar ticks and tick labels
cbar.set_ticks(np.linspace(0, 1, 3, endpoint=True))
cbar.set_ticklabels(['low', 'medium', 'high'])
plt.show()
