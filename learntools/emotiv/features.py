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

import numpy as np
import scipy.stats

import sys

def eig_corr(data, **kwargs):
    """Ported from https://github.com/drewabbot/kaggle-seizure-prediction/blob/master/qms/QT_eig_corr.m
    data: multichannel eeg data
    lambda: eigenvalues
    """
    C = np.corrcoef(data, rowvar=0)
    C[np.isnan(C)] = 0.0
    C[np.isinf(C)] = 0.0
    w, v = np.linalg.eig(C)
    return np.sort(w)

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
    lxchannels = abs(eig_corr(dspect))
    lxfreqbands = abs(eig_corr(dspect.T))

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

def all_features(*args, **kwargs):
    return (eig_corr(*args, **kwargs),) + frequency_bands(*args, **kwargs) + stat_moments(*args, **kwargs) + hjorth(*args, **kwargs)

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
        print eig_corr(data)

    from learntools.emotiv.data import load_raw_data
    ds = load_raw_data(eeg_dataset, conds=['PositiveHighArousalPictures', 'PositiveLowArousalPictures'])
    for row in ds['eeg']:
        print frequency_bands(row)
        print hjorth(row)
