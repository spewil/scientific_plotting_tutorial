"""Synthesis: tools for synthesizing spike trains and recordings

Uses recorded noise and clustered spikes.
"""

import numpy as np
from numpy.random import randn, rand, RandomState

from spike import Snapshot
import typing
from typing import Any, List, Optional, Tuple


def generate_random_spike_times(total_duration_s,
                                fs,
                                n_src,
                                on_dur_s,
                                off_dur_s,
                                on_off_sigma_pct,
                                avg_rate_hz,
                                rate_sigma_hz,
                                random_state: Optional[int] = None):
    """Synthesize on-off period for each motor unit and generate spike times

    Generate spike times for several sources (motor units) over time interval
    `total_duration_s`. Each source is synthesized independently.  Each source
    is on for `on_dur_s` and then off for `off_dur_s`.  The actual duration of
    each period is normally distributed, with the specified mean and a standard
    deviation that is defined as a percentage of that mean with
    `on_off_sigma_pct`. When a given source is "on", it generates spikes with
    an average rate of `avg_rate_hz` plus Gaussian noise with standard
    deviation `rate_sigma_hz`.

    Parameters
    ----------
    total_duration_s: total time (in seconds) of synthesis
    fs: sampling rate (in Hz)
    n_src: number of sources to synthesize
    on_dur_s: average duration (in seconds) of "on" segments
    off_dur_s: average duration (in seconds) of "off" segments
    on_off_sigma_pct: standard deviation of duration variability (in percent)
    avg_rate_hz: average firing rate during "on" segments (in Hz)
    rate_sigma_hz: standard deviation of rate variability (in Hz)

    Returns
    -------
    spike_times: (T x n_src)
        Array of spike amplitudes over time for each source. Sparse. T =
        round(fs * total_duration_s)
    """

    rs = np.random.RandomState(20)

    # Derived parameters in per-sample units
    T = int(fs * total_duration_s)
    on_dur = np.round(on_dur_s * fs)
    off_dur = np.round(off_dur_s * fs)

    spike_times = np.zeros((T, n_src))
    for src in range(n_src):
        stay_on_for = -1
        t = 0
        while t < T:
            if stay_on_for <= 0:
                stay_on_for = on_dur * (
                    1 + rs.randn() * on_off_sigma_pct / 100)
                if t == 0:
                    off_for = int(rand() * (on_dur + off_dur))
                else:
                    off_for = int(off_dur *
                                  (1 + randn() * on_off_sigma_pct / 100))
                t = t + off_for
            else:
                spike_times[t, src] = 1
                isi = int(fs / (avg_rate_hz + rs.randn() * rate_sigma_hz))
                t = t + isi
                stay_on_for = stay_on_for - isi
    return spike_times


def generate_ordered_spike_times(total_duration_s, fs, n_src, on_dur_s,
                                 off_dur_s, avg_rate_hz):
    """For debugging purposes, generate each source in turn

    Similar to `generate_random_spike_times`, but without randomness. Triggers
    each source in order and triggers spikes within each "on" interval at the
    average rate.

    Parameters
    ----------
    total_duration_s: total time (in seconds) of synthesis
    fs: sampling rate (in Hz)
    n_src: number of sources to synthesize
    on_dur_s: average duration (in seconds) of "on" segments
    off_dur_s: average duration (in seconds) of "off" segments
    avg_rate_hz: average firing rate during "on" segments (in Hz)

    Returns
    -------
    spike_times: (T x n_src)
        Array of spike amplitudes over time for each source. Sparse. T =
        round(fs * total_duration_s)
    """

    # Derived parameters in per-sample units
    T = fs * total_duration_s
    on_dur = np.round(on_dur_s * fs)
    off_dur = np.round(off_dur_s * fs)

    spike_times = np.zeros((T, n_src))
    for src in range(n_src):
        stay_on_for = -1
        t = 0
        while t < T:
            if stay_on_for <= 0:
                stay_on_for = on_dur
                if t == 0:
                    off_for = int(src * (on_dur + off_dur) / n_src)
                else:
                    off_for = off_dur
                t = t + off_for
            else:
                spike_times[t, src] = 1
                isi = int(fs / avg_rate_hz)
                t = t + isi
                stay_on_for = stay_on_for - isi
    return spike_times


class NotEnoughNoiseError(Exception):
    pass


def add_noise(combined: np.ndarray, all_noise: np.ndarray) -> np.ndarray:
    """Add recorded noise to a synthetic recording

    Parameters
    ----------
    combined: (samples x channels)
        EMG recording, probably a combination of synthetic motor unit
        recordings
    all_noise: (samples2 x channels)
        EMG recording of noise. Sliced and diced to add to `combined`. Probably
        a quiescent period in a real EMG recording.
    """

    T = combined.shape[0]

    # TODO: loop noise
    if T <= all_noise.shape[0]:
        noise = all_noise[:T, :]
    else:
        raise NotEnoughNoiseError(f"Not enough noise: {T}")

    # Add extra noise
    final_signal = combined + noise
    return final_signal, noise


Electrode = Tuple[float, float]
Source = Tuple[float, float]


def lead_field(
        source: Source = (0., 0.),
        electrode: Electrode = (1., 0.),
) -> Any:
    """One element of the Lead Field matrix.

    Estimate the contribution of one source to one electrode.
    A very simple model is used, where the amplitude decay as the inverse of
    the distance.

    Parameters
    ----------
    source: tuple of floats
        the (x, y) position of the source.
    electrode: tuple of floats
        the (x, y) position of the electrode.

    Returns
    -------
    out: float
        the attenuation coefficient of the lead field matrix.
    """
    sr = 0.063
    Ka = 6.
    c = 1. / (2 * np.pi * sr)
    distance = (source[0] - electrode[0])**2 + (source[1] - electrode[1])**2

    out = c / np.sqrt(distance * Ka)
    return out


def lead_field_r2(
        source: Source = (0., 0.),
        electrode: Electrode = (1., 0.),
) -> Any:
    """One element of the Lead Field matrix.

    Estimate the contribution of one source to one electrode.
    A very simple model is used, where the amplitude decay as the inverse of
    the squared distance.

    Parameters
    ----------
    source: tuple of floats
        the (x, y) position of the source.
    electrode: tuple of floats
        the (x, y) position of the electrode.

    Returns
    -------
    out: float
        the attenuation coefficient of the lead field matrix.
    """
    sr = 0.063
    Ka = 6.
    c = 1. / (2 * np.pi * sr)
    distance = np.sqrt((source[0] - electrode[0])**2 +
                       (source[1] - electrode[1])**2)

    out = c / (np.power(distance, 2) * Ka)
    return out


def generate_mixing_model(
        n_elec: int = 16,
        n_groups: int = 10,
        n_mu_per_group: int = 10,
        depth: float = 8e-3,
        spread: float = 2e-3,
        radius: float = 3.5e-2,
        random_state: Optional[int] = None
) -> Tuple[List[Electrode], List[Source], np.ndarray]:
    """Generate spatial mixing model.

    Given a number of electrodes and sources + some extra parameters, generate
    a realistic spatial mixing model based on the signal propagations.

    Parameters
    ----------
    n_elec: int
        number of electrode. they will be equally spaced around the arm.
    n_groups: int
        number of groups of motor units. they will be equaly spaced around the
        arm.
    n_mu_per_group: int
        number of motor unit per group.
    depth: float
        depth (in meter) of the center of the groups.
    spread: float
        spread of the motor unit in the group.
    radius: float
        radius of the forearm, in meters.
    random_state: int
        random state seed.

    Returns
    -------
    electrodes: list or tuple of floats
        A list representing all the (x, y) position of the electrodes.
    sources: list of tuple of floats
        A list representing all the (x, y) position of the sources.
    A: np.ndarray
        the lead field matrix i.e. the spatial mixing model.
    """

    # generate electrodes positions
    electrodes = [(radius * np.cos(alpha), radius * np.sin(alpha))
                  for alpha in 2 * np.pi * np.arange(0, 1, 1. / n_elec)]

    # generate sources positions
    rs = np.random.RandomState(random_state)
    sources = []

    for ii, alpha in enumerate(2 * np.pi * np.arange(0, 1, 1. / n_groups)):
        # define the center of the group
        center = np.array([(radius - depth) * np.cos(alpha),
                           (radius - depth) * np.sin(alpha)])
        for jj in range(n_mu_per_group):
            pos = rs.randn(2) * spread + center
            # if outside the forearm diameter, regenerate a new one.
            while (np.sqrt((pos**2).sum()) > (radius * 0.95)):
                pos = rs.randn(2) * spread + center
            sources.append(pos)

    A = np.zeros((n_elec, len(sources)))

    for ii in range(n_elec):
        for jj in range(len(sources)):
            A[ii, jj] = lead_field(electrodes[ii], sources[jj])

    return electrodes, sources, A


def generate_single_mixing_model(
        n_elec: int = 16,
        source_position: Tuple = (8e-3, 0.),
        radius: float = 3.5e-2,
) -> np.ndarray:
    """Generate spatial mixing model for a single source.
    Given a number of electrodes and sources + some extra parameters, generate
    a realistic spatial mixing model based on the signal propagations.
    Parameters
    ----------
    n_elec: int
        number of electrode. they will be equally spaced around the arm.
    source_position: tuple of floats
        polar coordinates of the source with a small difference, instead
        of providing the magnitude of the vector, the depth is provided.
        essentially, (depth, theta) where depth is depth of source from
        the surface in m. Theta is angle wrt electrode 0.
    radius: float
        radius of the forearm, in meters.
    Returns
    -------
    A: np.ndarray
        the lead field matrix i.e. the spatial mixing model.
    """

    # generate electrodes positions
    electrodes = [(radius * np.cos(alpha), radius * np.sin(alpha))
                  for alpha in 2 * np.pi * np.arange(0, 1, 1. / n_elec)]

    depth, alpha = source_position

    center = np.array([(radius - depth) * np.cos(alpha),
                       (radius - depth) * np.sin(alpha)])

    A = np.zeros(n_elec)

    for ii in range(n_elec):
        A[ii] = lead_field_r2(electrodes[ii], center)

    return A


def action_potential_model(a1: float = 1.,
                           a2: float = 1.,
                           speed: float = 1.,
                           fs: float = 2000.,
                           duration: float = 0.02) -> np.ndarray:
    """Model on a action potential.

    The model is a 3 exponential model inspire from [1]_.

    It is parametrized by the relative amplitude of the positive and negative
    peak, the speed of the traveling wave, and the total amplitude of the AP.

    Parameters
    ----------
    a1: float
        relative amplitude of the fist peak
    a2: float
        relative amplitude of the second peak
    speed: float
        speed of the wave. this influence the time difference between the first
        and second peak. values lower than 1 increase speed, and higher
        decrease it.
    fs: float
        sampling rate
    duration: float
        duration, in second, of the action potential

    Returns
    -------
    spike: np.ndarray
        the action potential waveform, sampled at the provided sampling
        frequency.

    References
    ----------
    .. [1] Fleisher, S. M., M. Studer, and G. S. Moschytz. "Mathematical model
           of the single-fibre action potential." Medical and Biological
           Engineering and Computing 22, no. 5 (1984): 433-439.
           https://link.springer.com/article/10.1007/BF02447703
    """
    n_samples = int(duration * fs)
    t = (0.11) * np.arange(-n_samples // 2, n_samples // 2) / fs
    a = 0.05e-3
    s = 4 * a * speed
    alpha1 = a1 / np.abs(a1 + a2)
    alpha2 = a2 / np.abs(a1 + a2)
    spike = (alpha1 * np.exp(-(
        (t + s) / (4 * a))**2) - alpha2 * np.exp(-((t - s) / (4 * a))**2) +
             (alpha2 * alpha1) * np.exp(-((t - 2 * s) / (4 * a))**2))

    # Center biggest peak of resulting spike to agree with analysis
    t_max = abs(spike).argmax()
    delta_t = t_max - n_samples // 2
    if delta_t > 0:
        spike = np.concatenate((spike[delta_t:], np.zeros(delta_t)))
    elif delta_t < 0:
        spike = np.concatenate((np.zeros(-delta_t), spike[:delta_t]))

    return spike


def generate_random_spikes(n_spikes: int = 20,
                           random_state: Optional[int] = None,
                           fs: float = 2000.,
                           duration: float = 0.02) -> List[np.ndarray]:
    """generate a list of spikes.


    Parameters
    ----------
    n_spikes: int
        number of different spikes to generate
    random_state: int, None
        random seed.
    fs: float
        sampling rate
    duration: float
        duration, in second, of the action potential

    Returns
    -------
    spike: list of np.ndarray
        A list of action piotential waveforms.
    """
    spikes = []
    rs = np.random.RandomState(random_state)
    for _ in range(n_spikes):
        sign = np.sign(rs.randn())
        a1 = rs.rand()
        a2 = rs.rand()
        speed = rs.randint(75, 125) / 100.

        spike = action_potential_model(
            a1=a1, a2=a2, speed=speed, fs=fs, duration=duration)

        amplitude = sign * rs.randint(50, 150) / 100.
        spikes.append(amplitude * spike)

    return spikes


def generate_emg(
        spike_templates: np.ndarray,
        spike_trains: np.ndarray,
        noise_sigma: float = 0.,
        spike_amplitude_sigma: float = 0.,
        rs: Optional[RandomState] = None,
) -> np.ndarray:
    """Convolves the spike templates with the spike trains and sums the
       results to return an emg. This method adds some noise on the convolved
       output, it also adds noise to the amplitude of each spike.

    Parameters
    ----------
    spike_templates : np.ndarray
        (n_spikes, n_template_samples, n_electrodes) shaped spike template.
    spike_trains : np.ndarray
        (n_spikes, n_samples) sized spike train.
    noise_sigma : float, optional
        Adds noise on each sample in uV, example 6.89*1e-6 (the default is 0.)
        noise is sampled from a 0 mean noise_sigma^2 normal distribution.
    spike_amplitude_noise : float, optional
        Adds variation to spike shapes by adding noise to each spike in the
        spike train, example 0.025 (the default is 0.)
        noise is sampled from a 0 mean spike_amplitude_sigma^2 normal
        distribution. Since spike amplitudes is 1, this can be considered
        a % fuzz of the spike.
    random_state : Optional[RandomState], optional
        Seed for the random number generator (the default is None)

    Returns
    -------
    np.ndarray
        (n_channels, n_samples) emg data, where n_samples is the spike train
        length from the spike_trains parameter.
    np.ndarray
        (n_spike_trains, n_samples) spike trains, with noise incorporated in
        it, this is the ground truth.
    """
    if rs is None:
        rs = RandomState(42)
    spike_trains = np.copy(spike_trains)

    n_spikes, n_samples, n_electrodes = spike_templates.shape
    n_spike_trains, n_samples = spike_trains.shape
    assert n_spikes == n_spike_trains

    emg = np.zeros((n_electrodes, n_samples))

    for spike_num in np.arange(n_spikes):
        train = spike_trains[spike_num, :]
        train[train == 1] += spike_amplitude_sigma * rs.randn(
            np.sum(train == 1))

    for electrode_num in np.arange(n_electrodes):
        for spike_num in np.arange(n_spikes):
            emg[electrode_num, :] += np.convolve(
                spike_templates[spike_num, :, electrode_num],
                spike_trains[spike_num],
                mode='same',
            )

    emg += rs.randn(n_electrodes, n_samples) * noise_sigma

    return emg, spike_trains


def generate_spike_template(
        n_electrodes: int = 16,
        radius_of_forearm: float = 3.5e-2,
        electrode: int = -1,
        source_position: Tuple = (8e-3, 0),
        a1: float = 0.3,
        a2: float = 0.7,
        speed: float = 1.0,
        amplitude: Optional[float] = 100 * 1e-6,
        sampling_hz: int = 2000,
        duration: float = 0.02,
) -> np.ndarray:
    """Generates an n_electrode spike template.

    Parameters
    ----------
    n_electrodes : int, optional
        Number of electrodes in the device (the default is 16)
    radius_of_forearm : float, optional
        Radius of the forearm in m (the default is 3.5e-2)
    electrode : int, optional
        The electrode on which the peak of the spike occurs (the default is -1)
        The electrode number spans from [0, 15].
    source_position : Tuple, optional
        The (depth, theta) coordinate of the spike source (m, theta)
        (the default is (8e-3, 0))
    a1 : float, optional
        Relative amplitude of first peak (the default is 0.3)
    a2 : float, optional
        Relative amplitude of second peak (the default is 0.7)
    speed : float, optional
        Conduction velocity see action potential model. (the default is 1.0)
    amplitude : float, optional
        Peak amplitude of the resultant spike uV. (the default is 100*1e-6)
    sampling_hz : int, optional
        Sampling Freq (the default is 2000)
    duration : float, optional
        Duration of the spike (the default is 0.02)

    Returns
    -------
    np.ndarray
        (n_samples, n_electrodes) spike template.
    """

    # Generate the action potential waveform
    ap = action_potential_model(a1, a2, speed, sampling_hz, duration)

    # Convert from electrode number to a source position
    if electrode >= 0 and electrode < n_electrodes:
        theta = (2 * np.pi / n_electrodes) * electrode
        depth = 8e-3
        source_position = (depth, theta)
    else:
        source_position = source_position

    # Generate the mixing model, i.e. the signal that would be received
    # at each electrode given the source position and the forearm radius.
    mixing = generate_single_mixing_model(n_electrodes, source_position,
                                          radius_of_forearm)

    # Multiply the action potential waveform by the mixing model to generate
    # the n_channel muap template.
    muap = ap[:, np.newaxis] * mixing[np.newaxis, :]

    # Ensure that the amplitudes are between [-1, 1]
    muap = muap / (np.abs(muap).max())

    muap = amplitude * muap

    return muap


def generate_random_spike_template(
        n_electrodes: int = 16,
        electrode: int = -1,
        source_position: Tuple = (8e-3, 0),
        amplitude: Optional[float] = None,
        rs: Optional[RandomState] = None,
) -> np.ndarray:
    """Generates a random spike template.

    Parameters
    ----------
    n_electrodes : int, optional
        Number of electrodes (the default is 16)
    electrode : int, optional
        Electrode on which we want to see the spike peaks (the default is -1)
    source_position : Tuple, optional
        (depth, theta) source position (m, rad) (the default is (8e-3, 0))
    amplitude : Optional[float], optional
        Amplitude of the resultant spike (the default is None)
    random_state : Optional[RandomState], optional
        Random number seed (the default is None)

    Returns
    -------
    np.ndarray
        (n_samples, n_electrodes) spike template.
    """

    if rs is None:
        rs = RandomState(42)

    sign = np.sign(rs.randn())
    a1 = rs.rand()
    a2 = rs.rand()
    speed = rs.randint(75, 125) / 100.
    if amplitude is None:
        amplitude = sign * rs.randint(50, 150) * 1e-6

    return generate_spike_template(
        n_electrodes=n_electrodes,
        electrode=electrode,
        source_position=source_position,
        a1=a1,
        a2=a2,
        speed=speed,
        amplitude=amplitude,
    )


class MotorUnitModel():
    """Model of a single Motor unit.

    This model is used to generate spike train for a given motor unit. A MU has
    a recruitment threshold (the level of force it start firering) and a
    maximum firering frequency.

    During the simulation, given a certain force level, the firering rate is
    estimated from a sigmoid curve centered around the activation threshold.
    From the firing rate, we then estimate the probability of the MU firing
    based on the last time it was triggered.

    Parameters
    ----------
    snapshot: Snapshot instance
        the snapshot object containing the action potential related to this
        motor unit.
    threshold: float
        the threshold for the motor unit recruitment, between 0 and 1.
    max_freq: int
        maximum firering rate
    fs: float
        sampling frequency.
    random_state:
        random seed.
    """

    def __init__(self,
                 snapshot: Snapshot,
                 threshold: float = 0.5,
                 max_freq: int = 30,
                 fs: float = 2000.,
                 random_state: Optional[int] = None) -> None:
        self.rs = np.random.RandomState(random_state)
        self.snapshot = snapshot
        self.count = 0
        self.threshold = threshold
        self.max_freq = max_freq
        self.fs = fs
        self.firing_jitter = 1. / (0.005 * fs)  # 5ms jitter

    def frequency(self, activity: float) -> Any:
        """spiking frequency."""
        if activity == 0:
            return 1e-12
        freq = self.max_freq / (1. + np.exp(-40 * (activity - self.threshold)))
        return freq

    def synthesize_recording(self, spike_times: np.ndarray) -> np.ndarray:
        """Synthesize recording.

        Synthesize recording given a spike train.

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
        return self.snapshot.synthesize_recording(spike_times)

    def update(self, activity: float) -> Any:
        """update motor units.

        For a given muscle activity level, return if the motor unit fire (1)
        or not (0).

        Parameters
        ----------
        activity: float
            the muscle activity, from 0 to 1.

        Returns
        -------
        trig: float
            return 1 if the motor unit fire, 0 otherwise.
        """
        # get spiking frequency
        freq = self.frequency(activity)
        isi = self.fs // freq

        # estimate proba of spiking
        self.count += 1
        pr = self.count / isi

        # if proba of spiking is above a threshold, we do a coin flip to
        # determine if the spike will fire. this is to add a bit of jitter on
        # the inter spike interval
        if pr > 0.95:
            # we use a binomial distribution with a proba of firing estimated
            # to have a 5ms jitter
            trig = self.rs.binomial(1, self.firing_jitter)
            if trig == 1:
                self.count = 0
        else:
            trig = 0.

        return trig


class MuscleModel():
    """Muscle Model.

    A muscle is a group of several motor units, ordered by recruitment
    threshold. for each of them, a MotorUnitModel will be instanciated, with
    uniformely spaced recruitment thresholds and random maximum spiking
    frequency between 10 and 30 Hz.

    Parameters
    ----------
    snapshots: list of Snapshot instance
        lsit of snapshot objects.
    random_state:
        random seed.
    fs: float
        sampling frequency.
    """

    def __init__(self,
                 snapshots: List[Snapshot],
                 random_state: Optional[int] = None,
                 fs: float = 2000.) -> None:
        self.rs = np.random.RandomState(random_state)

        self.motor_units = [
            MotorUnitModel(
                snapshot=snapshot,
                threshold=ii / len(snapshots),
                max_freq=self.rs.randint(10, 30),
                random_state=random_state,
                fs=fs) for ii, snapshot in enumerate(snapshots)
        ]

    def update(self, activity: float) -> np.ndarray:
        """update muscle.

        For a given sample of muscle activity level, return an array containing
        the status of each motor unit of the muscle.

        Parameters
        ----------
        activity: float
            the muscle activity, from 0 to 1.

        Returns
        -------
        spikes_sample: np.ndarray (n_units x 1)
            array of length `n_units`, corresponding to the status of each
            motor unit for the current time sample, with 1 if the motor unit
            fire, 0 otherwise.
        """
        return np.array([unit.update(activity) for unit in self.motor_units])

    def synthesize_recording(self, spike_times: np.ndarray) -> np.ndarray:
        """Synthesize recording.

        Synthesize recording given a spike train for each motor unit.

        Parameters
        ----------
        spike_times: np.ndarray (samples x n_units)
            vector indicating amplitude to trigger at each sample (not a list
            of trigger times).

        Returns
        -------
        recording: np.ndarray (samples x channels)
            simulated multi-channel recording
        """

        emg = self.motor_units[0].synthesize_recording(spike_times[:, 0])

        for ii in range(1, spike_times.shape[1]):
            emg += self.motor_units[ii].synthesize_recording(
                spike_times[:, ii])
        return emg

    def get_spike_trains(self, muscle_activity: np.ndarray) -> np.ndarray:
        """Get the spike train.

        From the muscle activity index, return a spike train for all motor
        units of the model.

        Parameters
        ----------
        muscle_activity: np.ndarray
            array of size (n_samples x 1) that correpond to the muscle
            activation index, from 0 to 1 (0 to 100% MVC).

        Returns
        -------
        spike_trains: np.ndarray
            array of shape (n_samples x n_units) representing
            the spiking activity of each MU over time.
        """

        return np.array([self.update(sample) for sample in muscle_activity])


class ForeArmModel():
    """Model of the forearm.

    This model allow a pseudo realistic synthesis of EMG signal.

    Parameters
    ----------
    n_electrodes: int
        number of electrodes in the model. they will be uniformely spaced
        around the arm.
    n_muscles: int
        number of muscles. they will be uniformely spaced around the arm.
    n_units_per_muscle: int
        Number of motor unit in the muscle.
    random_state: int or None
        random seed.
    fs: float
        sampling frequency.
    spike_duration: float
        the duration of the spikes in second.
    """

    def __init__(self,
                 n_electrodes: int = 16,
                 n_muscles: int = 14,
                 n_units_per_muscle: int = 20,
                 random_state: Optional[int] = None,
                 fs: float = 2000.,
                 spike_duration: float = 0.02) -> None:
        self.n_electrodes = n_electrodes
        self.n_muscles = n_muscles
        self.n_units_per_muscle = n_units_per_muscle
        self.rs = np.random.RandomState(random_state)
        self.fs = fs

        # get the mixing model
        model = generate_mixing_model(
            n_elec=n_electrodes,
            n_groups=n_muscles,
            n_mu_per_group=n_units_per_muscle,
            random_state=random_state)

        self.electrodes, self.sources, self.mixing = model

        # iterate over muscles
        self.muscles = []  # type: List[MuscleModel]
        for ii in range(n_muscles):

            # generate action potential waveforms
            if random_state is not None:
                rs = random_state + ii  # type: Optional[int]
            else:
                rs = None

            waveforms = generate_random_spikes(
                n_spikes=n_units_per_muscle,  # number of columns in the model
                random_state=rs,
                fs=fs,
                duration=spike_duration)

            # generate the snapshots
            snapshots = []
            for jj, waveform in enumerate(waveforms):
                mix = self.mixing[:, ii * n_units_per_muscle + jj]
                snapshots.append(Snapshot(np.outer(mix, waveform).T))

            # generate muscle model
            muscle = MuscleModel(snapshots, random_state=rs)
            self.muscles.append(muscle)

    def get_spike_trains(self, muscle_activity: np.ndarray) -> np.ndarray:
        """Get the spike train.

        From the muscle activity index, return a spike train for all motor
        units of the model.

        Parameters
        ----------
        muscle_activity: np.ndarray
            array of size (n_samples x n_muscles) that correpond to the muscle
            activation index, from 0 to 1 (0 to 100% MVC).

        Returns
        -------
        trains: np.ndarray
            3D array of shape (n_samples x n_muscles x n_units) representing
            the spiking activity of each MU for each muscle over time.
        """
        trains = np.zeros((len(muscle_activity), self.n_muscles,
                           self.n_units_per_muscle))

        for kk in range(muscle_activity.shape[1]):
            if muscle_activity[:, kk].sum() == 0:
                # skip if muscle never active
                continue

            trains[:, kk] = self.muscles[kk].get_spike_trains(
                muscle_activity[:, kk])

        return trains

    def generate_emg_signal(
            self,
            spike_trains: np.ndarray,
            amplification_noise: float = 6.89,
            spike_amplitude_noise: float = 0.025) -> np.ndarray:
        """Generate EMG from spike trains.

        Generate EMG from a given set of spike trains. A white noise will be
        added to the EMG data.
        A small noise on the amplitude of the spike will be added to simulate
        variability in spikes amplitude.

        Parameters
        ----------
        spike_trains: np.ndarray, shape (n_samples x n_muscles x n_units)
            the spike trains, as generated by `get_spike_trains`
        amplification_noise: float
            the standard deviation of the amplification noise, in uV RMS.
            Typically around 7 for the King and Lefferts bands.
        spike_amplitude_noise: float
            standard deviation of the noise on the amplitude of the spike.
            Noise is added to the trigger of the spike trains, so that it
            simulate variation on the amplitude of the spikes.
            This value is typically low. Avoid value larger than 0.2 as it can
            lead to invert the sign of the spike.

        Returns
        -------
        emg: np.ndarray (n_samples x n_electrodes)
            the synthesized EMG signal.
        """
        nt, nc = len(spike_trains), self.n_electrodes
        emg = np.zeros((nt, nc))

        for ii, muscle in enumerate(self.muscles):
            train = spike_trains[:, ii]
            if np.sum(train) == 0:
                # skip if no spike in the train
                continue

            # add a little jitter on the spike amplitude
            train[train == 1] += spike_amplitude_noise * self.rs.randn(
                np.sum(train == 1))
            emg += muscle.synthesize_recording(train)

        emg += self.rs.randn(nt, nc) * amplification_noise
        return emg

    def plot_model(self):
        """plot physical model."""
        from pybmi.modeling.spike_sorting import viz
        return viz.plot_model(self.sources, self.electrodes)
