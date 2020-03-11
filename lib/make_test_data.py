import pickle
import numpy as np

NUM_DATAPOINTS = 1000
NUM_CHANNELS = 3

data = {}
data["x"] = np.arange(NUM_DATAPOINTS).reshape(NUM_DATAPOINTS, 1)
data["y"] = 0.1 * np.random.normal(size=(
    NUM_DATAPOINTS, NUM_CHANNELS)) + np.arange(1, NUM_CHANNELS + 1).reshape(
        1, NUM_CHANNELS)

with open('pre_session_data.pickle', 'wb') as f:
    pickle.dump(data, f)
