from __future__ import print_function, division

from datetime import datetime
import itertools
import cPickle as pickle
import glob
import numpy as np
import os
import traceback

from learntools.data import Dataset
from learntools.data.dataset import LISTEN_TIME_FORMAT
from learntools.libs.utils import normalize_table, loadmat
from learntools.libs.eeg import signal_to_freq_bins

# headers used in the raw data taken directly from matlab
RAW_HEADERS = [
    ('subject', Dataset.STR),
    ('eeg_sequence', Dataset.SEQFLOAT),
    ('condition', Dataset.SEQINT), # TODO is this the column type we really want?
    ('time', Dataset.TIME),
]

# headers used in the raw data that has been segmented
SEGMENTED_HEADERS = [
    ('subject', Dataset.ENUM), # subject id
    ('source', Dataset.STR), # denotes the file and eeg index where the segment was taken from
    ('eeg', Dataset.MATFLOAT), # fixed duration due to truncation for uniformity
    ('condition', Dataset.ENUM), # only one label characterizes the sequence
]


ACTIVITY_CONDITIONS = {
    'EyesOpen': 1,
    'EyesClosed': 2,
    'rest': 3,
    'NegativeHighArousalPictures': 4,
    'NeutralLowArousalPictures': 5,
    'PositiveHighArousalPictures': 6,
    'NegativeLowArousalPictures': 7,
    'PositiveLowArousalPictures': 8,
    'EroticHighArousalPictures': 9,
    'CountBackwards': 10,
    'Ruminate': 11,
    'DrawStickFigure': 12,
}

META_CONDITIONS = {
    'BEGIN': 132,
    'END': 136,
    'INNER': 0,
}

CONDITIONS = dict(ACTIVITY_CONDITIONS.items() + META_CONDITIONS.items())
CONDITIONS_STR = dict((v, k) for k, v in CONDITIONS.items())


def prepare_data(dataset_name, conds=None, **kwargs):
    """load siegle data into a Dataset

    Args:
        conds (string[], optional): list of conditions that we want the dataset to contain
            e.g. ['EyesClosed', 'EyesOpen']. If not provided, all conditions will be loaded

    Returns:
        Dataset: a dataset with the following columns:
            group: the source of the data (contains descriptives about the subject, etc.)
            condition: the task that the subject was doing while the eeg was being recorded
            eeg: eeg feature vector for each task
    """
    bands = ['theta', 'alpha', 'beta', 'gamma']
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    eeg_headers = [('{0}_{1}'.format(*x), Dataset.FLOAT) for x in
                   itertools.product(bands, channels)]
    headers = [('fname', Dataset.STR), ('Condition', Dataset.ENUM)] + eeg_headers
    data = Dataset.from_csv(dataset_name, headers)
    data.rename_column('fname', 'group')
    data.rename_column('Condition', 'condition')
    data.set_column('eeg', Dataset.MATFLOAT)
    for i, eeg in enumerate(itertools.izip(*[data[h] for (h, _) in eeg_headers])):
        data.get_column('eeg')[i] = eeg
    data.get_column('eeg').data = normalize_table(data['eeg'])

    # only keep selected conditions
    # TODO: turn this these temporary mode switches into a context
    if conds is not None:
        all_conds = data.orig['condition']  # TODO: follow up on possible deadlock caused
        # when I put data['condition'] in the filter
        selected = filter(lambda i: all_conds[i] in conds, xrange(data.n_rows))
        data.reorder(selected)
        cond_data = data.orig['condition']
        data.set_column('condition', Dataset.ENUM)  # reset the condition column
        for i, c in enumerate(cond_data):
            data.get_column('condition')[i] = c
    return data


def convert_raw_data(directory, output):
    raw_files = glob.glob(os.path.join(directory, '*.mat'))

    subjects = {}

    n_rows = len(raw_files)
    ds = Dataset(RAW_HEADERS, n_rows)

    for i, raw_filename in enumerate(raw_files):
        print(raw_filename)
        try:
            raw_file = loadmat(raw_filename)
            filename, extension = os.path.splitext(os.path.basename(raw_filename))
            p = raw_file['p']
            eeg_sequence = p['EEG']
            condition = p['OtherData'][2, :]
            dt = datetime(*tuple(p['hdr']['orig']['T0']))
            timestr = dt.strftime(LISTEN_TIME_FORMAT)

            ds[i] = (filename, eeg_sequence, condition, timestr)
        except Exception as e:
            traceback.print_exc()
            raise e

    print(len(raw_files), "files loaded")

    with open(output, 'wb') as f:
        pickle.dump(ds.to_pickle(), f, protocol=pickle.HIGHEST_PROTOCOL)


def _segment_gen(segment_idx, segment_cond):
    """Generator for segment (begin, end, label). Coalesces cascading segments with the same label.
    segment_idx: numpy array for the indices for the beginning of each label
    segment_cond: numpy array the condition started by index"""

    length = len(segment_idx)
    if length == 0:
        raise StopIteration

    # iterate through segments, coalescing when possible
    i = 0
    while i < length - 2:
        # find the segment end
        j = i + 1
        cond = segment_cond[i]
        while j < length - 1 and segment_cond[j] == cond:
            j += 1

        yield segment_idx[i], segment_idx[j], cond
        i = j


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

    subjects = []
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
        if CONDITIONS['BEGIN'] not in segment_cond or CONDITIONS['END'] not in segment_cond:
            print("begin and end not in eeg sequence", subject)
            continue

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
            # TODO: do something about this
            segment_samples = segment_end - segment_begin

            if segment_samples < duration * sample_rate:
                print(source, "with length", segment_samples, "too small")
                continue
            else:
                # truncate sample to duration seconds
                segment_end = min(segment_end, segment_begin + duration * sample_rate)
                # shape should be (duration * sample_rate) by eeg vector length
                eeg_segment = eeg_seq[segment_begin:segment_end, :]

                # get subject id or generate one
                if subject not in subjects:
                    subjects.append(subject)
                    subject_id = subjects.index(subject)
                else:
                    subject_id = subjects.index(subject)

                # Fourier transform on eeg
                # Window size of 1 s, overlap by 0.5 s
                eeg_freqs = []

                if 'finer_freq_bins' in kwargs:
                    cutoffs = [2.0 ** (i * 0.5) for i in xrange(10)]
                else:
                    cutoffs = [0.5, 4.0, 7.0, 12.0, 30.0]

                for i in (x * 0.5 for x in xrange(duration * 2)):
                    # window is half second duration (in samples) by eeg vector length
                    window = eeg_segment[int(i * sample_rate/2) : int((i + 1) * sample_rate/2)]
                    # there are len(cutoffs)-1 bins, window_freq is a list of will have a frequency vector of num channels
                    window_freq = signal_to_freq_bins(window, cutoffs=cutoffs, sampling_rate=128.0)

                    eeg_freqs.append(np.concatenate(window_freq))

                if 'larger_intervals' in kwargs:
                    for i in (x * 2.5 for x in xrange(duration // 4)):
                        window = eeg_segment[int(i * sample_rate * 2.5) : int((i + 1) * sample_rate * 2.5)]
                        eeg_freqs.append(np.concatenate(signal_to_freq_bins(window, cutoffs=cutoffs, sampling_rate=128.0)))

                # (num windows * num bins) * num channels
                eeg_freqs = np.concatenate(eeg_freqs)

                # Flatten into a vector
                eeg_freqs_flattened = np.ravel(eeg_freqs)

                segments.append((subject_id, source, eeg_freqs_flattened, label))

    # add all segments to the new dataset
    new_ds = Dataset(SEGMENTED_HEADERS, len(segments))
    for i, seg_data in enumerate(segments):
        new_ds[i] = seg_data

    # TODO normalization should be shared across some columns
    new_ds.get_column('eeg').data = normalize_table(new_ds['eeg'])

    return new_ds
