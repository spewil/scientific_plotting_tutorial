import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import interp1d

NUM_SAMPLES = 10000

x = np.arange(NUM_SAMPLES)
xp = np.linspace(0, NUM_SAMPLES, 10, endpoint=True)
fp = np.ones(xp.shape) * 30
# speed = np.interp(x=x, xp=xp, fp=fp).reshape(1, -1)
fp += np.random.normal(size=(len(fp)), scale=15)
fp[0] = 0

speed = interp1d(xp, fp, kind='cubic')(x).reshape(1, -1)

noise = np.random.normal(size=(1, NUM_SAMPLES), scale=2)

speed += noise
print(speed.shape)

with open("behavior.pickle", "wb") as f:
    pickle.dump(speed, f)

# plt.plot(speed[0, :])
# plt.show()
