import numpy as np

from learntools.kt.data import prepare_new_data2, prepare_new_data
from learntools.libs.utils_for_test import use_logger_in_test


# TODO: this test is out of date. Replace it with a better one
'''
@use_logger_in_test
def test_prepare_dataset():
    dataset_name = 'data/data4.gz'
    ds, train_idx, valid_idx = prepare_new_data2(dataset_name, top_n=14, cv_fold=0)
    prepared_data = prepare_new_data(dataset_name, top_eeg_n=14, eeg_only=1, cv_fold=0)
    subject_x, skill_x, correct_y, start_x, eeg_x, eeg_table, stim_pairs, train_idx, valid_idx = prepared_data
    assert all(subject_x == ds['subject'])
    assert all(correct_y == ds['correct'])
    assert all(start_x == ds['start_time'])
    np.allclose(eeg_table[eeg_x], ds['eeg'])
'''
