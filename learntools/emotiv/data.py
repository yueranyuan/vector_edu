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
from learntools.libs.wavelet import signal_to_wavelet

DEFAULT_CALIBRATION_FILE_LOCATION = 'raw_data/allcalibqualityreport.csv'

# headers used in the raw data taken directly from matlab
RAW_HEADERS = [
    ('subject', Dataset.STR),
    ('eeg_sequence', Dataset.SEQFLOAT),
    ('condition', Dataset.SEQINT),
    ('time', Dataset.TIME),
]

# headers used in the raw data that has been segmented
SEGMENTED_HEADERS = [
    ('subject', Dataset.ENUM),  # subject id
    ('source', Dataset.STR),  # denotes the file and eeg index where the segment was taken from
    ('eeg', Dataset.SEQFLOAT),  # fixed duration due to truncation for uniformity
    ('condition', Dataset.ENUM),  # only one label characterizes the sequence
]

# headers used in the segmented data that has been labeled with features
FEATURED_HEADERS = [
    ('subject', Dataset.ENUM),  # subject id
    ('source', Dataset.STR),  # denotes the file and eeg index where the segment was taken from
    ('eeg', Dataset.MATFLOAT),  # fixed duration due to truncation for uniformity
    ('condition', Dataset.ENUM),  # only one label characterizes the sequence
]
# siegle's data
SIEGLE_HEADERS = [
    ('eeg', Dataset.MATFLOAT),
    ('condition', Dataset.ENUM),
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


def load_siegle_data(dataset_name, conds=None, **kwargs):
    if conds is None:
        conds = ACTIVITY_CONDITIONS.keys()
    cond_values = [ACTIVITY_CONDITIONS[cond] for cond in conds]
    f = loadmat(dataset_name)
    M = f['M']
    n_rows = len(M)
    # data is sorted by cond
    np.random.shuffle(M)
    Xs = M[:, 3:]
    ys = M[:, 1]

    idxs = np.arange(n_rows)[reduce(np.logical_or, [ys == ACTIVITY_CONDITIONS[cond] for cond in conds])]

    ds = Dataset(SIEGLE_HEADERS, len(idxs))
    for i in xrange(len(idxs)):
        ds[i] = (Xs[idxs[i]], ys[idxs[i]])

    return ds


def convert_raw_data(directory, output):
    raw_files = glob.glob(os.path.join(directory, '*.mat'))

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

    return new_ds


class FeatureGenerationException(Exception):
    pass


def _gen_featured_dataset(ds, func, *args, **kwargs):
    """Helper function that applies 'func' to generate features for the eeg segment of each row.

    Normalizes eeg features

    Args:
        ds (Dataset): segmented dataset
        func (function): feature transformation function to apply to each row

    Returns:
        Dataset: featured dataset
    """
    featured_rows = []
    for i in xrange(len(ds)):
        _, source, eeg_segment, _ = ds[i]
        subject, _, _, label = ds.orig[i]
        try:
            eeg_features = func(eeg_segment, *args, **kwargs)
        except FeatureGenerationException:
            # TODO: convert to a warning
            print('could not generate features for row {i}, (subject: {subject}, source: {source}, label: {label})'.format(
                i=i,
                subject=subject,
                source=source,
                label=label
            ))
            continue
        featured_rows.append((subject, source, eeg_features, label))

    # add all rows to the new dataset
    new_ds = Dataset(FEATURED_HEADERS, len(featured_rows))
    for i, row in enumerate(featured_rows):
        new_ds[i] = row

    # TODO normalization should be shared across some columns
    print('feature vector width: {}'.format(new_ds.get_column('eeg').data.shape[1]))  # TODO: replace with 'width' once we merge
    new_ds.get_column('eeg').data = normalize_table(new_ds['eeg'])

    return new_ds


def gen_fft_features(ds, duration=10, sample_rate=128, cutoffs=None):
    if cutoffs is None:
        cutoffs = [0.5, 4.0, 7.0, 12.0, 30.0]

    def _fft_eeg_segment(eeg_segment, duration, sample_rate, cutoffs):
        # Fourier transform on eeg
        # Window size of 1 s, overlap by 0.5 s
        eeg_freqs = []

        for i in (x * 0.5 for x in xrange(duration * 2)):
            # window is half second duration (in samples) by eeg vector length
            window = eeg_segment[int(i * sample_rate/2) : int((i + 1) * sample_rate/2)]
            # there are len(cutoffs)-1 bins, window_freq is a list of will have a frequency vector of num channels
            window_freq = signal_to_freq_bins(window, cutoffs=cutoffs, sampling_rate=sample_rate)

            eeg_freqs.append(np.concatenate(window_freq))

        # (num windows * num bins) * num channels
        eeg_freqs = np.concatenate(eeg_freqs)
        return eeg_freqs

    return _gen_featured_dataset(ds, _fft_eeg_segment, duration=duration, sample_rate=sample_rate, cutoffs=cutoffs)


def gen_wavelet_features(ds, duration=10, sample_rate=128, depth=5, min_length=3, max_length=14, family='db6'):
    def _wavelet_eeg_segment(eeg_segment, duration, sample_rate, depth, min_length, max_length, family):
        # cut eeg to desired length (so that all wavelets are the same length)
        desired_length = duration * sample_rate
        if len(eeg_segment) < desired_length:
            raise FeatureGenerationException("signal not long enough")
        eeg_segment = eeg_segment[:desired_length]

        # wavelet transform
        eeg_wavelets = []
        for i in xrange(eeg_segment.shape[1]):
            eeg_wavelet = signal_to_wavelet(eeg_segment[:, i], min_length=min_length, max_length=max_length,
                                            depth=depth, family=family)
            eeg_wavelets += eeg_wavelet

        return np.concatenate(eeg_wavelets)

    return _gen_featured_dataset(ds, _wavelet_eeg_segment, duration=duration, sample_rate=sample_rate,
                                 depth=depth, min_length=min_length, max_length=max_length, family=family)

def filter_indices_by_condition(dataset, idx, conds):
    mapping = dict(dataset['condition'].ienum_pairs)
    idx = np.array(idx)
    # convert from cond string to cond enum, to internal cond enum, to mask
    want = [dataset.get_data('condition')[idx] == mapping[ACTIVITY_CONDITIONS[cond]] for cond in conds]
    return idx[reduce(np.logical_or, want)]