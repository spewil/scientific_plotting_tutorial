import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
"""
Demo 1: Introducition to the matplotlib plt function

Here I provide a minimum example of using matplotlib for plotting.
matplotlib.pyplot takes a minimum of 1 argument,
corresponding to the y coordinate values of the points you want to plot.


"""

with open('data/pre_session_data.pickle', 'rb') as f:
    data2 = pickle.load(f)
datashape = data2["y"].shape
data2["y"] += np.ones(datashape)
plt.plot(data2["x"], data2["y"])
plt.show()
