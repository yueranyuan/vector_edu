import numpy as np
import pywt
import matplotlib.pyplot as plt
import cPickle as pickle

from learntools.data import Dataset
from learntools.emotiv.data import SEGMENTED_HEADERS, ACTIVITY_CONDITIONS, _segment_gen
from learntools.libs.eeg import signal_to_freq_bins
from learntools.emotiv.data import segment_raw_data as wavelet_raw_data


def segment_raw_data(dataset_name, conds=None, duration=10, sample_rate=128, **kwargs):
    """Loads raw siegle data from a pickled Dataset and extracts sequences of
    eeg vectors which have a single known label.

    Args:
        ds: the raw Dataset
        duration: duration the segment should be, in seconds
        sample_rate: the number of eeg vectors per second
    Returns:
        Dataset
    """
    with open(dataset_name, 'rb') as f:
        ds = Dataset.from_pickle(pickle.load(f))

    segments = []

    conds_values = [ACTIVITY_CONDITIONS[k] for k in conds] if conds is not None else ACTIVITY_CONDITIONS.values()

    # extract segments of relevant activity from the eeg vector sequence
    for i in xrange(len(ds)):
        subject, eeg_seq, condition_seq, rec_time = ds[i]

        # find the nonzero elements of condition_seq, which are the actual labels
        segment_idx = np.nonzero(condition_seq)[0]
        segment_cond = condition_seq[segment_idx]

        # The beginning/ending of a recording is denoted by 132 and 136, respectively;
        # only use the sequence if they exist
        #if CONDITIONS['BEGIN'] not in segment_cond or CONDITIONS['END'] not in segment_cond:
        #    print("begin and end not in eeg sequence", subject)
        #    continue

        # The beginning of a segment is denoted by that label; inside the segment,
        # the label is 0; the end is denoted by another label.
        for segment_begin, segment_end, label in _segment_gen(segment_idx, segment_cond):
            source = "{}@{}".format(subject, segment_begin)

            # don't use segment if the label is unknown
            if label not in ACTIVITY_CONDITIONS.values():
                continue

            # filter out unwanted labels
            if label not in conds_values:
                continue

            # if segment is too short, don't use
            segment_samples = segment_end - segment_begin

            if segment_samples < duration * sample_rate:
                print('{0} with length {1} too small'.format(source, segment_samples))
                continue
            else:
                # truncate sample to duration seconds
                segment_end = min(segment_end, segment_begin + duration * sample_rate)
                # shape should be (duration * sample_rate) by eeg vector length
                eeg_segment = eeg_seq[segment_begin:segment_end, :]

                # clip any outlier segments
                # take things within 25th-75th percentile
                sorted_eeg_segment = np.sort(eeg_segment, axis=0)[len(eeg_segment) / 4 : len(eeg_segment) * 3 / 4]
                mean = np.mean(sorted_eeg_segment, axis=0)
                std = np.std(sorted_eeg_segment - mean, axis=0)
                lo_thresh = mean - 4 * std
                hi_thresh = mean + 4 * std
                eeg_segment = np.minimum(np.maximum(lo_thresh, eeg_segment), hi_thresh)
                segments.append((subject, source, eeg_segment, label))

    # add all segments to the new dataset
    new_ds = Dataset(SEGMENTED_HEADERS, len(segments))
    for i, seg_data in enumerate(segments):
        new_ds[i] = seg_data

    #new_ds = gen_fft_features(new_ds, duration=duration, sample_rate=sample_rate)
    #new_ds = gen_wavelet_features(new_ds, duration=duration, sample_rate=sample_rate)

    return new_ds

'''
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
'''


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

if __name__ == "__main__":
    from itertools import compress
    dataset_name = 'data/raw_seigle.gz'
    max_length = 4
    ds = wavelet_raw_data(dataset_name, conds=['EyesOpen', 'EyesClosed'], wavelet=True, wavelet_max_length=max_length)
    eeg = ds['eeg'][:]
    eeg = eeg.reshape((eeg.shape[0], 14, 6, max_length))
    eeg_no_time = np.mean(eeg, axis=3)
    plot_conditions(eeg=eeg_no_time, conditions=ds['condition'])
