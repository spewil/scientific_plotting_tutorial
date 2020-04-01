import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("../data/spike_trains.pickle", "rb") as f:
    spike_trains = pickle.load(f)

raster_matrix = np.zeros(shape=(len(spike_trains), spike_trains[0].shape[1]))
for i, train in enumerate(spike_trains):
    raster_matrix[i, :] = train[0]

raster_matrix[raster_matrix > 0.5] = 1

np.save("../data/raster_matrix.npy", raster_matrix)

fig = plt.figure()
main_ax = fig.add_subplot(111)
main_ax.spines['top'].set_color('none')
main_ax.spines['bottom'].set_color('none')
main_ax.spines['left'].set_color('none')
main_ax.spines['right'].set_color('none')
main_ax.tick_params(
    labelcolor='w', top=False, bottom=False, left=False, right=False)
main_ax.set_ylabel("Units")
main_ax.grid(True, which='both', axis='both')
axes = [
    fig.add_subplot(raster_matrix.shape[0], 1, i + 1)
    for i in range(raster_matrix.shape[0])
]

for i, (raster, ax) in enumerate(zip(raster_matrix[::-1], axes)):
    ax.eventplot(np.nonzero(raster), color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([1])
    ax.set_yticklabels([i + 1])
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_xticks(list(range(0, 10001, 1000)))
    ax.patch.set_visible(False)
ax.set_xticklabels(list(range(0, 10001, 1000)), rotation=30)
plt.show()
