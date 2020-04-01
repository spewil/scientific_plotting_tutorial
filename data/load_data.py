import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

## fake treadmill
with open('behavior.pickle', 'rb') as f:
    behavior_data = np.array(pickle.load(f)[0])

print(behavior_data.shape)
plt.figure()
plt.plot(behavior_data)

## fake spikes
with open('ephys.pickle', 'rb') as f:
    ephys = np.array(pickle.load(f)).reshape(20, -1)

print(ephys.shape)
plt.figure()
for i in range(2):
    plt.plot(ephys[i])

# fake spike raster
spike_raster = np.load('raster_matrix.npy')
print(spike_raster.shape)
plt.figure()
plt.eventplot(np.where(spike_raster[0]))

plt.show()
