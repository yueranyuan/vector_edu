"""
Features.

- Binned fft (frequency_bands)
- Shannon entropy across the same band (frequency_bands)
- Shannon entropy across band pairs (frequency_bands)
- Spectrum correlation across band pairs (eig_corr, frequency_bands)
- Time-series correlation matrix and its eigenvalues (eig_corr, frequency_bands)
- Statistical moments: variance, skewness and kurtosis (stat_moments)
- Spectral edge power of 50% power up to 40 Hz (frequency_bands)
- Hjorth parameters: activity, mobility, and complexity (hjorth)

TODO:
- Fractal dimensions
"""

from learntools.libs.eeg import signal_to_freq_bins
from learntools.libs.wavelet import signal_to_wavelet

import numpy as np
import scipy.stats

import sys


class FeatureGenerationException(Exception):
    pass


def _eig_corr(data, **kwargs):
    """Ported from https://github.com/drewabbot/kaggle-seizure-prediction/blob/master/qms/QT_eig_corr.m
    data: multichannel eeg data
    lambda: eigenvalues
    """
    C = np.corrcoef(data, rowvar=0)
    C[np.isnan(C)] = 0.0
    C[np.isinf(C)] = 0.0
    w, v = np.linalg.eig(C)
    return np.sort(w)


def eig_corr(data, **kwargs):
    return (_eig_corr(data, **kwargs),)


def frequency_bands(data, sfreq=128, tfreq=40, ppow=0.5, bands=(0.1, 4.0, 8.0, 12.0, 30.0, 64.0), **kwargs):
    """Ported from https://github.com/drewabbot/kaggle-seizure-prediction/blob/master/qms/QT_6_freq_bands.m
    data: multichannel eeg data
    sfreq: sampling frequency
    tfreq: cutoff frequency to calculate spectral power, tfreq < sfreq/2
    ppow: percentage of power up to tfreq used to calculate spectral edge power, 0 < ppow < 1
    bands: frequency levels in Hz
    dspect: spectrum of each frequency band
    spentropy_channels: spectral entropy across same band
    spentropy_bands: spectral entropy across band pairs
    spedge: spectral edge power of 50% power up to 40 Hz
    lxchannels: eigenvalues of spectral correlation matrix between channels
    lxfreqbands: eigenvalues of spectral correlation matrix between frequency bands
    """
    D = abs(np.fft.fft(data, axis=0))       # take fft of each channel
    D[0, :] = 0                             # set DC component to 0
    D = D / D.sum(axis=0, keepdims=True)    # normalize each channel

    # find number of data points corresponding to frequency bands
    l = np.array(bands)
    lseg = np.round(l / sfreq * data.shape[0])  # segments corresponding to frequency bands

    # power spectrum at each frequency band
    dspect = np.array([2 * np.sum(D[lseg[n] : lseg[n + 1], :], axis=0) for n in xrange(len(l) - 1)])

    # Shannon entropy
    spentropy_channels = -np.sum(dspect * np.log(dspect), axis=0)
    spentropy_bands = -np.sum(dspect.T * np.log(dspect.T), axis=0)

    # spectral edge frequency
    topfreq = np.round(float(data.shape[0]) / sfreq * tfreq)
    A = np.cumsum(D[:topfreq, :], axis=0)
    B = A - A.max(axis=0) * ppow
    spedge = abs(B).argmin(axis=0) / topfreq * tfreq
    
    # eigenvalues of spectral correlation matrix
    lxchannels = abs(_eig_corr(dspect))
    lxfreqbands = abs(_eig_corr(dspect.T))

    return (dspect, spentropy_channels, spentropy_bands, spedge, lxchannels, lxfreqbands)


def stat_moments(data, **kwargs):
    """Ported from https://github.com/drewabbot/kaggle-seizure-prediction/blob/master/qms/QT_statistical_moments.m
    data: multichannel eeg data
    m: mean
    v: variance
    s: skewness
    k: kurtosis
    """

    m = np.mean(data, axis=0)
    data2 = data - m
    v = np.var(data2, axis=0)
    s = scipy.stats.skew(data2)
    k = scipy.stats.kurtosis(data2)

    return (m, v, s, k)


def hjorth(data, **kwargs):
    """https://notendur.hi.is/steinng/qeegeliability07.pdf section 2.3.2
    data: multichannel eeg data
    activity: a0
    mobility: sqrt(a1/a0)
    complexity: sqrt(a2/a1 - a1/a0)
    a0: variance of signal
    a1: variance of first derivative of signal
    a2: variance of second derivative of signal
    """
    a0 = np.var(data, axis=0)
    s1 = np.diff(data, axis=0)
    a1 = np.var(s1, axis=0)
    s2 = np.diff(s1, axis=0)
    a2 = np.var(s2, axis=0)
    activity = a0
    mobility = np.sqrt(a1 / a0)
    complexity = np.sqrt(a2 / a1 - a1 / a0)

    return (activity, mobility, complexity)


def windowed_fft_variable(data, duration=10, sample_rate=128, cutoffs=(0.5, 4.0, 7.0, 12.0, 30.0), fft_window=1.5, **kwargs):
    # Fourier transform on eeg
    # Window size of 1 s, overlap by 0.5 s
    eeg_freqs = []
    overlap = 0.5
    start_t = 0.0
    data = data[:duration * sample_rate]
    while (start_t + fft_window) * sample_rate < len(data):
        end_t = start_t + fft_window
        window = data[int(start_t * sample_rate):int(end_t * sample_rate)]
        start_t = end_t - overlap

        # there are len(cutoffs)-1 bins, window_freq is a list of will have a frequency vector of num channels
        window_freq = signal_to_freq_bins(window, cutoffs=cutoffs, sampling_rate=sample_rate)

        eeg_freqs.append(np.concatenate(window_freq))

    # (num windows * num bins) * num channels
    eeg_freqs = np.concatenate(eeg_freqs)
    return (eeg_freqs,)


def windowed_fft(data, duration=10, sample_rate=128, cutoffs=(0.5, 4.0, 7.0, 12.0, 30.0), **kwargs):
    """
    data: multichannel eeg data
    duration: length of window
    sample_rate: sample rate
    cutoffs: fft bin thresholds
    """
    # Fourier transform on eeg
    # Window size of 1 s, overlap by 0.5 s
    eeg_freqs = []

    for i in (x * 0.5 for x in xrange(duration)):
        # window is half second duration (in samples) by eeg vector length
        window = data[int(i * sample_rate) : int((i + 1) * sample_rate)]
        # there are len(cutoffs)-1 bins, window_freq is a list of will have a frequency vector of num channels
        window_freq = signal_to_freq_bins(window, cutoffs=cutoffs, sampling_rate=sample_rate)

        eeg_freqs.append(np.concatenate(window_freq))

    # (num windows * num bins) * num channels
    eeg_freqs = np.concatenate(eeg_freqs)
    return (eeg_freqs,)


def just_fft(data, duration=10, sample_rate=128, cutoffs=(0.5, 4.0, 7.0, 12.0, 30.0), **kwargs):
    eeg_freqs = np.concatenate(signal_to_freq_bins(data, cutoffs=cutoffs, sampling_rate=sample_rate))
    print(eeg_freqs)
    return (eeg_freqs,)


def wavelet(data, duration=10, sample_rate=128, depth=0, min_length=10, max_length=None, family='db2', **kwargs):
    # cut eeg to desired length (so that all wavelets are the same length)
    desired_length = duration * sample_rate
    if len(data) < desired_length:
        raise FeatureGenerationException("signal not long enough")
    data = data[:desired_length]

    # wavelet transform
    eeg_wavelets = []
    for i in xrange(data.shape[1]):
        eeg_wavelet = signal_to_wavelet(data[:, i], min_length=min_length, max_length=max_length,
                                        depth=depth, family=family)
        eeg_wavelets += eeg_wavelet

    return (np.concatenate(eeg_wavelets),)


def raw(data, lower_boundary=0.05, upper_boundary=0.95, **kwargs):
    N = data.shape[0]
    boundaries = (N * lower_boundary, N * upper_boundary)
    return (np.concatenate(data[boundaries[0]:boundaries[1], :].T), )


FEATURE_MAP = {
    'eig_corr': eig_corr,
    'frequency_bands': frequency_bands,
    'stat_moments': stat_moments,
    'hjorth': hjorth,
    'windowed_fft': just_fft,
    'wavelet': wavelet,
    'raw': raw
}


def construct_feature_generator(feature_strs):
    """Returns a feature function (which returns a flattened feature vector).
    feature_strs: a list of strings representing features
    """
    feature_fns = [FEATURE_MAP[s] for s in feature_strs]

    def _feature_generator(data, *args, **kwargs):
        features = [np.concatenate([subfeature.ravel() for subfeature in feature_fn(data, *args, **kwargs)]) for feature_fn in feature_fns]
        return np.concatenate(features)

    return _feature_generator


if __name__ == '__main__':
    eeg_dataset = sys.argv[1] # pass in converted raw data path
    anscombe_quartet = (
        np.array([[10.0, 8.04], [8.0, 6.95], [13.0,  7.58], [9.0, 8.81], [11.0, 8.33], [14.0, 9.96], [6.0, 7.24], [ 4.0,  4.26], [12.0, 10.84], [7.0, 4.82], [5.0, 5.68]]),
        np.array([[10.0, 9.14], [8.0, 8.14], [13.0,  8.74], [9.0, 8.77], [11.0, 9.26], [14.0, 8.10], [6.0, 6.13], [ 4.0,  3.10], [12.0,  9.13], [7.0, 7.26], [5.0, 4.74]]),
        np.array([[10.0, 7.46], [8.0, 6.77], [13.0, 12.74], [9.0, 7.11], [11.0, 7.81], [14.0, 8.84], [6.0, 6.08], [ 4.0,  5.39], [12.0,  8.15], [7.0, 6.42], [5.0, 5.73]]),
        np.array([[ 8.0, 6.58], [8.0, 5.76], [ 8.0,  7.71], [8.0, 8.84], [8.0,  8.47], [8.0,  7.04], [8.0, 5.25], [19.0, 12.50], [ 8.0,  5.56], [8.0, 7.91], [8.0, 6.89]]),
    )

    for data in anscombe_quartet:
        print stat_moments(data)

    from learntools.emotiv.data import load_raw_data
    ds = load_raw_data(eeg_dataset, conds=['PositiveHighArousalPictures', 'PositiveLowArousalPictures'])
    for row in ds['eeg']:
        print frequency_bands(row)
        print hjorth(row)
