import cPickle
import gzip
from itertools import compress

import numpy as np

from learntools.data import Dataset


def test_load():
    dataset_name = 'data/data4.gz'
    with gzip.open(dataset_name, 'rb') as f:
        subject_x, skill_x, correct_y, start_x, eeg_x, stim_pairs = cPickle.load(f)
    from learntools.kt.data import old_gz_to_dataset
    dataset = old_gz_to_dataset(dataset_name)

    eeg_mask = np.not_equal(eeg_x, None)

    def _mask(column):
        return list(compress(column, eeg_mask))
    assert all(dataset.get_data('subject') == _mask(subject_x))
    assert all(dataset.get_data('correct') == _mask(correct_y))
    assert all(dataset.get_data('start_time') == _mask(start_x))
    assert np.all(dataset.get_data('eeg') == _mask(eeg_x))

    dataset.mode = Dataset.ORIGINAL
    stim_dict = {v: k for (k, v) in stim_pairs}
    skill_orig = [stim_dict[s] for s in skill_x]
    assert dataset.get_data('skill') == list(compress(skill_orig, eeg_mask))


def test_new_align():
    with gzip.open('data/data4.gz', 'rb') as f:
        subject_x, skill_x, correct_y, start_x, eeg_x, stim_pairs = cPickle.load(f)
