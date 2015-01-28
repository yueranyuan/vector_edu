from itertools import izip
import random

import numpy as np

from learntools.libs.common_test_utils import use_logger_in_test
from learntools.data import cv_split
from learntools.emotiv.base import BaseEmotiv
from learntools.data import Dataset


def gen_small_emotiv_data():
    conds = ['cond1', 'cond2', 'cond1', 'cond2', 'cond1']
    eegs = [[1, 2, 3, 4],
            [4, 3, 2, 1],
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [1, 2, 3, 4]]
    headers = [('condition', Dataset.ENUM), ('eeg', Dataset.MATFLOAT)]
    dataset = Dataset(headers=headers, n_rows=len(conds))
    for i, row in enumerate(izip(conds, eegs)):
        dataset[i] = row
    return dataset


def gen_random_emotiv_data():
    eeg_width = 60
    source_magnitude = 0.5
    noise_magnitude = 1.5
    n_rows = 8000
    eeg_sources = {'cond1': np.random.random(eeg_width) * source_magnitude,
                   'cond2': np.random.random(eeg_width) * source_magnitude}
    conds = ['cond1', 'cond2']
    headers = [('condition', Dataset.ENUM), ('eeg', Dataset.MATFLOAT)]
    dataset = Dataset(headers=headers, n_rows=n_rows)
    for i in xrange(n_rows):
        cond = conds[random.randint(0, 1)]
        eeg = eeg_sources[cond] + np.random.random(eeg_width) * noise_magnitude
        dataset[i] = (cond, eeg)
    return dataset


@use_logger_in_test
def test_emotive_base():
    dataset = gen_small_emotiv_data()
    train_idx, valid_idx = cv_split(dataset, percent=0.3, fold_index=0)
    prepared_data = (dataset, train_idx, valid_idx)

    model = BaseEmotiv(prepared_data, batch_size=1)
    best_loss, best_epoch = model.train_full(n_epochs=40, patience=40)
    assert best_loss > 0.8


@use_logger_in_test
def test_emotive_base_random():
    dataset = gen_random_emotiv_data()
    train_idx, valid_idx = cv_split(dataset, percent=0.2, fold_index=0)
    prepared_data = (dataset, train_idx, valid_idx)

    model = BaseEmotiv(prepared_data, batch_size=50)
    best_loss, best_epoch = model.train_full(n_epochs=100, patience=100)
    assert best_loss > 0.8
