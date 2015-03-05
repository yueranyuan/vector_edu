import numpy as np
import pywt
import matplotlib.pyplot as plt
import cPickle as pickle

from learntools.data import Dataset
from learntools.emotiv.data import SEGMENTED_HEADERS, ACTIVITY_CONDITIONS, _segment_gen
from learntools.libs.eeg import signal_to_freq_bins


dataset_name = 'raw_data/partial.pickle'

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
                segments.append((subject, source, eeg_segment, label))

    # add all segments to the new dataset
    new_ds = Dataset(SEGMENTED_HEADERS, len(segments))
    for i, seg_data in enumerate(segments):
        new_ds[i] = seg_data

    #new_ds = gen_fft_features(new_ds, duration=duration, sample_rate=sample_rate)
    #new_ds = gen_wavelet_features(new_ds, duration=duration, sample_rate=sample_rate)

    return new_ds

ds = segment_raw_data(dataset_name, conds=['EyesOpen', 'EyesClosed'])

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