import numpy as np
import matplotlib.pyplot as plt

from learntools.emotiv.data import segment_raw_data, gen_wavelet_features
from learntools.emotiv.filter import filter_data
from learntools.libs.wavelet import signal_to_wavelet


def show_raw_wave(eeg):
    for channel in xrange(14):
        plt.plot(eeg[:, channel])
    plt.show()


def show_raw_specgram(eeg, label, block=False):
    fig, axs = plt.subplots(nrows=14, ncols=1)
    for channel in xrange(14):
        #axs[channel].plot(signal_to_freq_bins(eeg[:, channel], cutoffs=[0.5, 4.0, 7.0, 12.0, 30.0], sampling_rate=128))
        axs[channel].specgram(eeg[:, channel], Fs=128)
        axs[channel].set_title("{}[{}]".format(label, channel))

    fig.show()
    if block:
        fig.ginput(timeout=0)
        plt.close('all')


def specgram_slideshow(ds):
    for row in xrange(len(ds)):
        show_raw_specgram(ds['eeg'][row], "cond=" + str(ds['condition'][row]), block=True)


def plot_conditions(eeg, conditions):
    eeg1_full = np.asarray(list(compress(eeg, conditions == 0)))
    eeg2_full = np.asarray(list(compress(eeg, conditions == 1)))

    # draw select trials
    for i in xrange(10):
        plt.subplot(1, 10, i + 1)
        plt.pcolor(eeg1_full[i], cmap=plt.cm.Blues)
    plt.show()

    eeg1 = np.mean(eeg1_full, axis=0)
    eeg2 = np.mean(eeg2_full, axis=0)

    def _plot_heatmap(data):
        return plt.pcolor(data, cmap=plt.cm.Blues)

    # draw between class difference
    plt.subplot(1, 3, 1)
    _plot_heatmap(eeg1)
    plt.subplot(1, 3, 2)
    _plot_heatmap(eeg2)
    plt.subplot(1, 3, 3)
    _plot_heatmap(eeg1-eeg2)
    plt.show()

    # draw within class difference
    plt.subplot(1, 4, 1)
    _plot_heatmap(np.mean(eeg1_full[:(len(eeg1) / 2)], axis=0))
    plt.subplot(1, 4, 2)
    _plot_heatmap(np.mean(eeg1_full[(len(eeg1) / 2):], axis=0))
    plt.subplot(1, 4, 3)
    _plot_heatmap(np.mean(eeg2_full[:(len(eeg2) / 2)], axis=0))
    plt.subplot(1, 4, 4)
    _plot_heatmap(np.mean(eeg2_full[(len(eeg2) / 2):], axis=0))
    plt.show()


def _shape(ys):
    """ Get the shape of a non-numpy python array. This assumes the first index of every dimension is
    indicative of the shape of the whole matrix.

    Examples:
        >>> _shape([1, 2, 3])
        [3]
        >>> _shape([[1, 2, 3], [4, 5]])
        [2, 3]
    """
    if hasattr(ys, '__len__'):
        return [len(ys)] + _shape(ys[0])
    else:
        return []


def plot_waves(ys, ylim=None):
    shape = _shape(ys)

    if len(shape) > 3:
        from operator import __mul__
        dim1 = reduce(__mul__, shape[:-2])
        dim2 = shape[-2]
    elif len(shape) == 3:
        dim1, dim2 = shape[:2]
    elif len(shape) == 2:
        dim1, dim2 = shape[0], 1
    elif len(shape) == 1:
        dim1 = dim2 = 1
    else:
        raise Exception("malformed ys")

    def _plot_wave(y, i):
        if len(_shape(y)) == 1:
            print i
            plt.subplot(dim1, dim2, i)
            if ylim is not None:
                plt.ylim(ylim)
            plt.plot(y)
            return i + 1
        else:
            for _y in y:
                i = _plot_wave(_y, i)
        return i

    _plot_wave(ys, 1)
    plt.show()


def analyze_waves(ds, n=20, ylim=(-80, 80)):
    for i in xrange(n):
        eeg_segment = ds['eeg'][i]
        wavelet = signal_to_wavelet(eeg_segment[:, 0], min_length=0, max_length=None,
                                    depth=5, family='db6')
        plot_waves(eeg_segment.T)
        plot_waves([(w, _downsample(w, 6)) for w in wavelet], ylim=ylim)
    exit()


def analyze_features(ds, max_length=4):
    ds = gen_wavelet_features(ds, duration=10, sample_rate=128, depth=5, min_length=3, max_length=max_length,
                              family='db6')
    filter_data(ds)

    eeg = ds['eeg'][:]
    eeg = eeg.reshape((eeg.shape[0], 14, 6, max_length))
    eeg_no_time = np.mean(eeg, axis=3)
    plot_conditions(eeg=eeg_no_time, conditions=ds['condition'])


if __name__ == "__main__":
    from itertools import compress
    from learntools.libs.wavelet import _downsample
    dataset_name = 'data/emotiv_all.gz'
    ds = segment_raw_data(dataset_name=dataset_name, conds=['EyesOpen', 'EyesClosed'])
    # analyze_waves(ds, n=2)
    analyze_features(ds, max_length=4)