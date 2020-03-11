import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 1000)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x, y1, "-b", label="sine")
ax.plot(x, y2, "-r", label="cosine")
ax.legend(loc="upper left", n)
ax.set_ylim(-1.5, 2.0)
plt.show()
