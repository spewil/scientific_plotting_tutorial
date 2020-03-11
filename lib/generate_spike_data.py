import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pickle
from synthesis import *

# simulate spike-sorted data

emg_data_data = []
spike_train_data = []

num_channels = 20
for ch in range(1, num_channels + 1):
    spike_templates = generate_random_spikes(n_spikes=1)
    spike_times = generate_random_spike_times(
        total_duration_s=5.,
        fs=2000,
        n_src=1,
        on_dur_s=5,
        off_dur_s=0,
        on_off_sigma_pct=1,
        avg_rate_hz=20,
        rate_sigma_hz=5).T
    num_spike_timepoints = len(spike_templates[0])
    spike_templates = np.array(spike_templates).reshape(
        1, num_spike_timepoints, 1)
    emg_data, spike_trains = generate_emg(
        spike_templates=spike_templates,
        spike_trains=spike_times,
        noise_sigma=0.01,
        spike_amplitude_sigma=0.01)
    plt.plot(emg_data[0, :], label=str(ch))

    emg_data_data.append(emg_data)
    spike_train_data.append(spike_trains)

    # with open("ephys.pickle", "wb") as f:
    #     pickle.dump(emg_data_data, f)

    # with open("spike_trains.pickle", "wb") as f:
    #     pickle.dump(spike_train_data, f)
    print(emg_data.shape)

plt.show()
