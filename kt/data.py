import os
from operator import or_
from itertools import imap, compress
import gzip
import cPickle
from collections import defaultdict

import numpy as np

from libs.logger import log, log_me
from libs.utils import normalize_table, transpose, idx_to_mask

BNTSM_TIME_FORMAT = '%m/%d/%y %I:%M %p'


def to_lookup_table(x):
    mask = np.asarray([v is not None for v in x], dtype=bool)
    if not mask.any():
        raise Exception("can't create lookup table from no data")

    # create lookup table
    valid_idxs = np.nonzero(mask)[0]
    width = len(x[valid_idxs[0]])
    table = np.zeros((1 + len(valid_idxs), width))  # leave the first row for "None"
    for i, l in enumerate(compress(x, mask)):
        table[i + 1] = np.asarray(l)
    table[1:] = normalize_table(table[1:])
    table[0] = table[1:].mean(axis=0)  # set the "None" vector to the average of all vectors

    # create a way to index into lookup table
    idxs = np.zeros(len(x), dtype='int32')
    idxs[valid_idxs] = xrange(1, len(valid_idxs) + 1)

    return idxs, table


@log_me('... loading data')
def prepare_eeglrkt_data(dataset_name='data/eeglrkt.txt', cv_fold=0, **kwargs):
    from libs.loader import load
    eeg_headers = ('kc_med', 'kc_att', 'kc_raww', 'kc_delta', 'fconf',
                   'kc_alpha', 'kc_beta', 'kc_gamma')
    data, enum_dict, _ = load(
        dataset_name,
        numeric=['fluent'],
        time=['start_time'],
        numeric_float=eeg_headers,
        enum=['user', 'skill'],
        time_format=BNTSM_TIME_FORMAT)
    sorted_idxs, _ = transpose(sorted(enumerate(data['start_time']),
                                      key=lambda v: v[1]))
    N = len(sorted_idxs)
    subject_x = data['user'][sorted_idxs]
    skill_x = data['skill'][sorted_idxs]
    start_x = data['start_time'][sorted_idxs]
    correct_y = data['fluent'][sorted_idxs] - 1
    eeg = np.column_stack([data[eh] for eh in eeg_headers])[sorted_idxs, :]

    min_encounters = 2
    skill_count = defaultdict(list)
    for i in range(N):
        skill_count[(subject_x[i], skill_x[i])].append(i)
    bad_indices = []
    for k, v in skill_count.iteritems():
        if len(v) < min_encounters:
            bad_indices += v
    mask = np.logical_not(idx_to_mask(bad_indices, N))
    subject_x = subject_x[mask]
    skill_x = skill_x[mask]
    start_x = start_x[mask]
    correct_y = correct_y[mask]
    eeg = eeg[mask]

    eeg_x, eeg_table = to_lookup_table(eeg)
    stim_pairs = list(enum_dict['skill'].iteritems())
    sorted_subj = sorted(np.unique(subject_x),
                         key=lambda s: sum(np.equal(subject_x, s)))
    held_out_subj = cv_fold % (len(sorted_subj) - 4)
    valid_subj_mask = np.equal(subject_x, held_out_subj)
    log('subjects {} are held out'.format(held_out_subj), True)
    train_idx = np.nonzero(np.logical_not(valid_subj_mask))[0]
    valid_idx = np.nonzero(valid_subj_mask)[0]

    return (subject_x, skill_x, correct_y, start_x, eeg_x, eeg_table, stim_pairs, train_idx, valid_idx)


def prepare_fake_data():
    # specific data
    correct_y = np.asarray([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], dtype='int32')
    skill_x = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype='int32')
    subject_x = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype='int32')
    held_out_subj = 2

    # generate filler data
    N = len(correct_y)
    start_x = np.asarray(range(N))
    eeg_x = np.zeros((N, 4))
    stim_pairs = [(s, 'skill_{s}'.format(s=s)) for s in np.unique(skill_x)]

    # generate CV folds
    held_out = np.equal(subject_x, held_out_subj)
    valid_idx = np.nonzero(held_out)
    train_idx = np.nonzero(np.logical_not(held_out))
    return (subject_x, skill_x, correct_y, start_x, eeg_x, stim_pairs, train_idx, valid_idx)


def prepare_data(dataset_name, **kwargs):
    if os.path.splitext(dataset_name)[1] == '.txt':
        return prepare_eeglrkt_data(dataset_name, **kwargs)
    else:
        return prepare_new_data(dataset_name, **kwargs)


@log_me('...loading data')
def prepare_new_data(dataset_name, top_n=0, top_eeg_n=0, eeg_only=1, normalize=0, cv_fold=0, **kwargs):
    with gzip.open(dataset_name, 'rb') as f:
        subject_x, skill_x, correct_y, start_x, eeg_x, stim_pairs = cPickle.load(f)
    correct_y -= 1
    subjects = np.unique(subject_x)
    indexable_eeg = np.asarray(eeg_x)

    def row_count(subj):
        return sum(np.equal(subject_x, subj))

    def eeg_count(subj):
        arr = indexable_eeg[np.equal(subject_x, subj)]
        return sum(np.not_equal(arr, None))

    # select only the subjects that have enough data
    if top_n:
        subjects = sorted(subjects, key=row_count)[-top_n:]
    if top_eeg_n:
        subjects = sorted(subjects, key=eeg_count)[-top_eeg_n:]
    mask = reduce(or_, imap(lambda s: np.equal(subject_x, s), subjects))

    # normalize eegs
    eeg_mask = np.not_equal(indexable_eeg, None)
    if normalize:
        for s in subjects:
            subj_mask = np.equal(subject_x, s)
            subj_eeg_mask = subj_mask & eeg_mask
            table = np.array([list(l) for l in indexable_eeg[subj_eeg_mask]])
            table = normalize_table(table)
            idxs = np.nonzero(subj_eeg_mask)
            for i in range(len(idxs)):
                eeg_x[i] = table[i]

    # mask out unselected data
    if eeg_only:
        mask &= eeg_mask
    subject_x = subject_x[mask]
    skill_x = skill_x[mask]
    correct_y = correct_y[mask]
    start_x = start_x[mask]
    eeg_x = list(compress(indexable_eeg, mask))

    # break cv folds
    # valid_subj_mask = random_unique_subset(subject_x, .9)
    u_subjects = np.unique(subject_x)
    heldout_subject = u_subjects[cv_fold % len(u_subjects)]
    valid_subj_mask = np.equal(subject_x, heldout_subject)
    log('subjects {} are held out'.format(np.unique(subject_x[valid_subj_mask])), True)
    train_idx = np.nonzero(np.logical_not(valid_subj_mask))[0]
    valid_idx = np.nonzero(valid_subj_mask)[0]

    eeg_x, eeg_table = to_lookup_table(eeg_x)

    return (subject_x, skill_x, correct_y, start_x, eeg_x, eeg_table, stim_pairs, train_idx, valid_idx)
