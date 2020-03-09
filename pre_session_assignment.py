import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
"""
Demo 0



"""

with open('data/pre_session_data.pickle', 'rb') as f:
    data2 = pickle.load(f)
datashape = data2["y"].shape
data2["y"] += np.ones(datashape)
plt.plot(data2["x"], data2["y"])
plt.show()
