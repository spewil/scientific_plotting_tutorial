import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-dark')

fig = plt.figure(1)
ax = fig.add_subplot(211)
img_plot = ax.imshow(np.random.random((100, 100)), vmin=.5)
(x0, y0, w, h) = img_plot.get_window_extent()
print(x0, y0, w, h)
cax = fig.add_axes([x0, y0, 0.05, h])
cbar = fig.colorbar(img_plot, cax=cax)
plt.show()
