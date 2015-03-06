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


def plot_wave(y):
    plt.plot(y)
    plt.show()


if __name__ == "__main__":
    from itertools import compress
    dataset_name = 'data/raw_seigle.gz'
    max_length = 4
    ds = segment_raw_data(dataset_name=dataset_name, conds=['EyesOpen', 'EyesClosed'])

    eeg_segment = ds['eeg'][0]
    wavelet = signal_to_wavelet(eeg_segment[:, 0], min_length=0, max_length=None,
                                depth=5, family='db6')
    plot_wave(eeg_segment[:, 0])
    for w in wavelet:
        plot_wave(w)
    exit()

    ds = gen_wavelet_features(ds, duration=10, sample_rate=128, depth=5, min_length=3, max_length=max_length,
                              family='db6')
    filter_data(ds)

    eeg = ds['eeg'][:]
    eeg = eeg.reshape((eeg.shape[0], 14, 6, max_length))
    eeg_no_time = np.mean(eeg, axis=3)
    plot_conditions(eeg=eeg_no_time, conditions=ds['condition'])
