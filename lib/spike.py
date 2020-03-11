"""Spike: Object-oriented spike sorting code

Quick-Start
-----------

Use the functions at the end of this module to create and manipulate the
objects.

    from . import detection, spike_utils

    # Detect spikes
    peak_indices = detection.pick_peaks_all_chans(signal)

    # Compute inverse covariance matrix for MVDR filters
    _, inv_mix_covar = spike_utils.lagged_cov_and_inv(synthetic_signal)

    # Create Spike objects at peaks
    spikes = create_spikes(signal, peak_indices)

    # Cluste Spikes into SpikeCluster objects
    clusters = cluster_spikes(spikes)

    # Derive MVDR filters
    for cl in clusters:
        cl.derive_mvdr(inv_mix_covar)

    # Apply MVDR filters
    filtered = np.concatenate(
        [cl.mvdr.filter_signal(signal) for cl in clusters], axis=1)
"""
import numpy as np

from typing import Iterable, List, Optional, Sequence, Tuple, Union

# from sklearn.cluster import MiniBatchKMeans


class Snapshot(object):
    """A recording that has a spatial and temporal extent.

    Parameters
    ----------
    snapshot: np.ndarray (time x channels)
        The matrix of samples that will be saved in this snapshot
    """

    def __init__(self, snapshot: np.ndarray) -> None:
        self.snapshot = snapshot

    def get_snapshot(self) -> np.ndarray:
        """Return the raw data, in image form."""
        return self.snapshot

    def feature_for_clustering(self) -> np.ndarray:
        """Vectorize snapshot as representation for clustering"""
        return np.reshape(self.snapshot.T, -1)

    def get_half_win_size(self) -> float:
        """Get half the temporal extent of the snapshot (in samples).

        This is the distance from the center to the beginning and end.
        """
        return float(self.snapshot.shape[0] / 2)

    def get_number_of_channels(self) -> int:
        return int(self.snapshot.shape[1])

    def imshow(self, ax, **kwargs) -> None:
        """Draw to a matplotlib axes object"""
        ax.imshow(self.snapshot.T, **kwargs)

    def __repr__(self) -> str:
        w, h = self.snapshot.shape
        return (f"Snapshot {w} x {h}, values: {self.snapshot.min():0.2f} - "
                f"{self.snapshot.max():0.2f}")

    def derive_mvdr(self, inv_cov_mat) -> "Snapshot":
        """Derive the MVDR filter from myself and an inverse covariance matrix

        inv_cov_mat dimensions should keep adjacent channels next to each other
        with adjacent time steps further apart.  See
        `spike_utils.lagged_cov_and_inv`
        """
        n_t, n_c = self.snapshot.shape
        reshaped = np.reshape(self.snapshot, (n_t * n_c, 1))
        phi_h = inv_cov_mat @ reshaped
        mvdr_unwrapped = phi_h / (reshaped.T @ phi_h)
        mvdr = np.reshape(mvdr_unwrapped, (n_t, n_c))
        return Snapshot(mvdr)

    def derive_matched_filter(self) -> "Snapshot":
        """Derive matched filter from myself.
        """
        mf = self.snapshot
        mf = mf - mf.mean()
        mf = mf / (mf.flatten()**2).sum()
        return Snapshot(mf)

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply this snapshot as a filter to a signal.

        Correlate this snapshot with signal and sum across channels.

        Parameters
        ----------
        signal: np.ndarray (samples x channels)
            Signal to filter. Must have the same number of channels as this
            snapshot

        Returns
        -------
        filtered: np.ndarray (samples x 1)
            Output of filter
        """
        n_t, n_ch = self.snapshot.shape
        T, n_ch2 = signal.shape
        assert n_ch == n_ch2

        filtered = np.zeros((T, 1))
        filt_tmp = signal @ self.snapshot.T
        dt = int(np.ceil((n_t - 1) / 2))
        for t in range(n_t):
            filtered[max(dt - t, 0):T + dt - t, 0] += filt_tmp[max(t - dt, 0):
                                                               T - dt + t, t]
        return filtered

    def synthesize_recording(self, spike_times: np.ndarray) -> np.ndarray:
        """Generate a synthetic multichannel recording.

        Trigger this snapshot at the requested times.

        Parameters
        ----------
        spike_times: np.ndarray (samples x 1)
            vector indicating amplitude to trigger at each sample (not a list
            of trigger times).

        Returns
        -------
        recording: np.ndarray (samples x channels)
            simulated multi-channel recording
        """
        T = spike_times.shape[0]
        n_ch = self.get_number_of_channels()
        recording = np.zeros((T, n_ch))
        ir = self.snapshot
        offset = int(self.get_half_win_size())

        ts = np.nonzero(spike_times)[0]
        for t in ts:
            if t < offset:
                ir_part = ir[:t + offset, :]
                recording[:t + offset, :] += spike_times[t] * ir_part
            elif t < T - offset:
                recording[t - offset:t + offset, :] += spike_times[t] * ir
            else:
                ir_part = ir[:T - t + offset, :]
                recording[t - offset:T, :] += spike_times[t] * ir_part

        return recording


class Spike(Snapshot):
    """A snapshot with a single peak, optionally including trigger information

    Parameters
    ----------
    snapshot: np.ndarray (samples x channels)
        Recording of snapshot from which to extract a snapshot centered on
        trigger time
    trigger_chan: int, default -1
        Channel at which this Spike was triggered in the recording
    trigger_time: int, default -1
        Time (in samples) at which this Spike was triggered in the recording
    """

    def __init__(self,
                 snapshot: np.ndarray,
                 trigger_chan: int = -1,
                 trigger_time: int = -1) -> None:
        super().__init__(snapshot)
        self.trigger_time = trigger_time
        self.trigger_chan = trigger_chan

    def overlaps(self, other: "Spike") -> bool:
        """Returns true if this spike overlaps with other in time"""
        dt = abs(self.trigger_time - other.trigger_time)
        return (self.trigger_time >= 0 and other.trigger_time >= 0
                and dt < self.get_half_win_size())

    def get_center(self) -> Tuple[int, int, float]:
        """Find the largest absolute value in this snapshot

        Returns
        -------
        my_t: time index of the peak in this snapshot (between 0 and T-1)
        my_c: channel index of the peak in this snapshot (between 0 and C-1)
        my_peak: value of the peak, can be positive or negative
        """
        ind = abs(self.snapshot).argmax()
        my_t, my_c = np.unravel_index(ind, self.snapshot.shape)
        my_peak = self.snapshot[my_t, my_c]
        return my_t, my_c, my_peak

    def get_amplitude(self) -> float:
        snapmin = np.min(self.feature_for_clustering())
        snapmax = np.max(self.feature_for_clustering())
        return snapmax - snapmin

    def has_consistent_chan(self) -> bool:
        """Check that channel peak agrees with recorded channel and that
        time peak is in the middle.
        """
        _, max_c, _ = self.get_center()
        return self.trigger_chan >= 0 and max_c == self.trigger_chan

    def is_centered(self, max_dt=1) -> bool:
        """Return True if the peak of this snapshot is at its center in time.
        """
        my_t, _, _ = self.get_center()
        return bool(abs(my_t - self.get_half_win_size()) <= max_dt)

    def __lt__(self, other) -> int:
        """Comparison operator (less-than)

        Snapshots are sorted by spike polarity, then channel of peak, then
        time of peak
        """
        my_t, my_c, my_p = self.get_center()
        other_t, other_c, other_p = other.get_center()
        return (my_c, my_t, my_p) < (other_c, other_t, other_p)

    def __repr__(self) -> str:
        my_t, my_c, my_peak = self.get_center()
        at_str = ""
        if self.trigger_time >= 0 or self.trigger_chan >= 0:
            at_str = f"at ({self.trigger_time}, {self.trigger_chan}), "
        return (f"Spike {at_str}centered at ({my_t}, {my_c})={my_peak:0.2f}")


def spike_from_recording(trigger_time: int,
                         trigger_chan: int,
                         recording: np.ndarray,
                         feat_win: int = 20,
                         demean_each_channel=False) -> Spike:
    """Create a Spike object by extracting a segment of a recording

    Parameters
    ----------
    trigger_time: int
        Time (in samples) at which this Spike was triggered in the recording
    trigger_chan: int
        Channel at which this Spike was triggered in the recording
    recording: np.ndarray (samples x channels)
        Recording from which to extract a snapshot centered on trigger time
    feat_win: int, default 20
        Duration (in samples) before and after trigger to save in snapshot

    Returns
    -------
    spike: Spike object extracted from that recording at that time
    """
    snapshot = recording[trigger_time - feat_win:trigger_time + feat_win, :]

    if demean_each_channel:
        snapshot -= np.mean(snapshot, axis=0)

    return Spike(
        snapshot, trigger_chan=trigger_chan, trigger_time=trigger_time)


class EmptySpikeClusterError(Exception):
    """Caused by trying to make a SpikeCluster with no spikes"""
    pass


class SpikeCluster(object):
    """A group of Spikes

    Computes the average of the snapshots of the spikes, which represents this
    cluster for the purposes of filtering, etc.

    Parameters
    ----------
    spikes: Sequence of Spike objects to include in this cluster, must be
        non-empty, otherwise an EmptySpikeClusterError will be raised
    """

    def __init__(self, spikes: Sequence[Spike]) -> None:
        self.spikes = spikes
        self.center = Spike(self._compute_mean())
        self.mvdr: Optional[Snapshot] = None
        self.matched_filter: Optional[Snapshot] = None

    def __len__(self) -> int:
        """Number of spikes in this cluster"""
        return len(self.spikes)

    def compute_median(self) -> np.ndarray:
        """Average together all of the constituent spikes to create a
        representation of this cluster. An empty cluster raises an
        EmptySpikeClusterError.
        """
        if self.spikes:
            all_snapshots = [s.snapshot.flatten() for s in self.spikes]
            profiles = np.vstack(all_snapshots)
            return np.reshape(np.median(profiles, axis=0), (40, -1))
        raise EmptySpikeClusterError()

    def _compute_mean(self) -> np.ndarray:
        """Average together all of the constituent spikes to create a
        representation of this cluster. An empty cluster raises an
        EmptySpikeClusterError.
        """
        if self.spikes:
            return sum(s.snapshot for s in self.spikes) / len(self.spikes)
        raise EmptySpikeClusterError()

    def derive_mvdr(self, inv_cov_mat) -> None:
        """Derive the MVDR filter for the average of this cluster.

        Saves the filter to the `mvdr` attribute.
        """
        self.mvdr = self.center.derive_mvdr(inv_cov_mat)

    def derive_matched_filter(self) -> None:
        """Derive the matched filter for the average of this cluster.

        Saves the filter to the `matched_filter` attribute
        """
        self.matched_filter = self.center.derive_matched_filter()

    def remove_spikes_mismatched_peak(self, max_dt=1, max_dc=1,
                                      verbose=False) -> None:
        """Remove spikes from this cluster with different peaks.

        Compares peak of Spike and peak of SpikeCluster and removes Spike from
        SpikeCluster if the peaks are too different from one another in time,
        in channel, or in polarity.

        Parameters
        ----------
        max_dt: maximum allowable difference in time between peaks
        max_dc: maximum allowable difference in channel between peaks
        verbose: print how many spikes were kept
        """
        new_spikes = []
        my_t, my_c, my_p = self.get_center()
        for s in self.spikes:
            s_t, s_c, s_p = s.get_center()
            if ((my_p > 0) == (s_p > 0) and abs(s_t - my_t) <= max_dt
                    and abs(s_c - my_c) <= max_dc):
                new_spikes.append(s)
        if verbose:
            print(f"Kept {len(new_spikes)} of {len(self.spikes)} spikes")
        self.spikes = new_spikes
        self.center = Spike(self._compute_mean())

    def remove_spikes_mismatched_peaks_by_majority(self,
                                                   max_dc=1,
                                                   verbose=False) -> None:
        """Remove spikes from this cluster with different peaks.

        Compares peak of Spike and peak of SpikeCluster and removes Spike from
        SpikeCluster if the peaks are too different from one another in time,
        in channel, or in polarity.

        Parameters
        ----------
        max_dt: maximum allowable difference in time between peaks
        max_dc: maximum allowable difference in channel between peaks
        verbose: print how many spikes were kept
        """
        major_chan_spikes = []
        result_spikes = []

        p_count = np.zeros(2, dtype=int)
        c_count = np.zeros(16, dtype=int)

        # get major channel spikes
        for spike in self.spikes:
            s_t, s_c, s_p = spike.get_center()
            c_count[s_c] += 1

        major_c = np.argmax(c_count)

        for s in self.spikes:
            s_t, s_c, s_p = s.get_center()
            if abs(s_c - major_c) <= max_dc:
                major_chan_spikes.append(s)

        # get major polarity
        for spike in major_chan_spikes:
            s_t, s_c, s_p = spike.get_center()
            if s_p > 0:
                p_count[1] += 1
            else:
                p_count[0] += 1

        major_p = np.argmax(p_count)

        for spike in major_chan_spikes:
            if ((major_p > 0) == (s_p > 0)):
                result_spikes.append(spike)

        if verbose:
            print('major polarity: {}, major channel: {}'.format(
                major_p, major_c))
            print(f"Kept {len(result_spikes)} of {len(self.spikes)} spikes")

        self.spikes = result_spikes
        self.center = Spike(self._compute_mean())

    def is_centered(self) -> bool:
        """Return True if the peak of this snapshot is at its center in time.
        """
        return self.center.is_centered()

    def get_half_win_size(self) -> float:
        """Get half the temporal extent of the snapshot (in samples).

        This is the distance from the center to the beginning and end.
        """
        return self.center.get_half_win_size()

    def get_center(self) -> Tuple[int, int, float]:
        """Find the largest absolute value in the center Snapshot

        Returns
        -------
        my_t: time index of the peak in this snapshot (between 0 and T-1)
        my_c: channel index of the peak in this snapshot (between 0 and C-1)
        my_peak: value of the peak, can be positive or negative
        """
        return self.center.get_center()

    def imshow(self, ax, **kwargs) -> None:
        """Draw to a matplotlib axes object"""
        self.center.imshow(ax, **kwargs)

    def synthesize_recording(self, spike_times: np.ndarray) -> np.ndarray:
        """Generate a synthetic multichannel recording.

        Trigger this snapshot at the requested times.

        Parameters
        ----------
        spike_times: np.ndarray (samples x 1)
            vector indicating amplitude to trigger at each sample (not a list
            of trigger times).

        Returns
        -------
        recording: np.ndarray (samples x channels)
            simulated multi-channel recording
        """
        return self.center.synthesize_recording(spike_times)

    def __repr__(self) -> str:
        return "Cluster centered at " + self.center.__repr__()

    def __lt__(self, other) -> int:
        return self.center.__lt__(other.center)


def create_spikes(emg: np.ndarray,
                  peak_indices: Iterable[Iterable[int]],
                  half_win: int = 20,
                  must_appear_alone: bool = True,
                  verbose: bool = False) -> List[Spike]:
    """Create Spike objects that are good for clustering and peak picking

    Removes Spikes that have a peak that is not consistent with their
    triggering or that occur too close to (within half_win samples of) another
    Spike.

    Parameters
    ----------
    emg: np.ndarray (samples x channels)
        recording to extract Spikes from
    peak_indices: Interable of peak times, for example from a function in the
        `detection` module
    half_win: size of half-window to use for Spikes. half_win=20 means the
        Spikes will have a temporal extent of 40, 20 forward in time, 20 back
    must_appear_alone: enforce that the detected channel and time must have the
        biggest peak in the observed region and that no other detections
        overlap it. Useful for initial clustering and peeling. Not necessary
        for reverse-correlation.
    verbose: print out count of spikes after each stage

    Returns
    -------
    spikes: List of Spike objects, at most one per entry in peak_indices, but
        possibly fewer
    """

    # Create spikes
    spikes = []
    for ch, mxi in enumerate(peak_indices):
        spikes.extend([
            spike_from_recording(
                m, ch, emg, half_win, demean_each_channel=False) for m in mxi
            if half_win < m < emg.shape[0] - half_win
        ])

    if verbose:
        print(f"Start with {len(spikes)} of size {half_win}")

    if must_appear_alone:

        # Keep spikes consistent with their triggering
        spikes = [
            s for s in spikes if s.is_centered() and s.has_consistent_chan()
        ]

        if verbose:
            print(f"Spikes after center-consistency has {len(spikes)}")

        # Sort spikes to find those that overlap with their neighbors
        spikes.sort(key=lambda x: x.trigger_time)

        if len(spikes) < 2:
            return spikes

        # Remove spikes that overlap with other spikes
        new_spikes = []
        if not spikes[0].overlaps(spikes[1]):
            new_spikes.append(spikes[0])
        for i in range(1, len(spikes) - 1):
            if not spikes[i].overlaps(
                    spikes[i - 1]) and not spikes[i].overlaps(spikes[i + 1]):
                new_spikes.append(spikes[i])
        if not spikes[-2].overlaps(spikes[-1]):
            new_spikes.append(spikes[-1])
        spikes = new_spikes

    if verbose:
        print(f"Final spikes has {len(spikes)}")

    return spikes


# def cluster_spikes(spikes: Iterable[Spike],
#                    n_clust: int = 200,
#                    min_clust_size: int = 10,
#                    random_state: Union[int, None] = None
#                    ) -> List[SpikeCluster]:
#     """Create SpikeClusters and only keep those that are well behavedself.

#     Does clustering using minibatch k-means.  Spikes are represented by their
#     `feature_for_clustering`.  Removes spikes from clusters that disagree on
#     peak location or polarity. Removes SpikeClusters that have too few Spikes.

#     Parameters
#     ----------
#     spikes: Iterable of Spike objects to be grouped into clusters
#     n_clust: (Maximum) number of clusters to use
#     min_clust_size: Minimum number of Spikes that need to be in each
#         SpikeCluster
#     """
#     min_clust_size = max([min_clust_size, 0])

#     # Get features for clustering from each spike
#     profiles = np.stack([s.feature_for_clustering() for s in spikes])

#     # Cluster
#     labels = MiniBatchKMeans(n_clusters=n_clust,
#                              random_state=random_state).fit_predict(profiles)

#     # Create actual SpikeCluster objects
#     spike_sets: List[List[Spike]] = [[] for c in range(n_clust)]
#     for s, lab in zip(spikes, labels):
#         spike_sets[lab].append(s)

#     clusters = [
#         SpikeCluster(ss) for ss in spike_sets if len(ss) >= min_clust_size
#     ]

#     # Remove spikes from clusters if they don't agree on peak location
#     for c in clusters:
#         try:
#             c.remove_spikes_mismatched_peak(verbose=True)
#         except EmptySpikeClusterError:
#             # This means that all of the spikes were removed, leaving the
#             # cluster in an invalid state, but it will be removed in the next
#             # step.
#             pass

#     # Ensure clusters are big enough still
#     clusters = [c for c in clusters if len(c) >= min_clust_size]

#     return sorted(clusters)


class FilterNotDerivedError(Exception):
    pass


def derive_mvdr_and_filter_signal(clusters: Iterable[SpikeCluster],
                                  emg: np.ndarray,
                                  inv_mix_covar: np.ndarray) -> np.ndarray:
    """Derive MVDR filters from cluster centers and apply them to a signal.

    Note that this modifies SpikeClusters to contain (updated) MVDR filters

    Parameters
    ----------
    clusters: Iterable of SpikeClusters to derive filters for and filter with
    emg: (samples x channels) signal to filter
    inv_mix_covar: inverse covariance matrix to use to derive MVDR filters. For
        example from `spike_utils.lagged_cov_and_inv`

    Returns
    -------
    filtered: np.ndarray (samples x clusters) output of all filters
    """

    # Derive MVDR filters
    for cl in clusters:
        cl.derive_mvdr(inv_mix_covar)

    # This is for mypy, derive_mvdr ensures that mvdr is not None...
    if cl.mvdr is None:
        raise FilterNotDerivedError()

    # Filter the signal
    filtered = np.concatenate(
        [cl.mvdr.filter_signal(emg) for cl in clusters], axis=1)

    return filtered
