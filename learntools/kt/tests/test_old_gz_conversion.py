import cPickle
import gzip
from itertools import compress

import numpy as np

from learntools.data import Dataset
from learntools.libs.common_test_utils import use_logger_in_test


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

    stim_dict = {v: k for (k, v) in stim_pairs}
    skill_orig = [stim_dict[s] for s in skill_x]
    assert dataset.orig['skill'] == list(compress(skill_orig, eeg_mask))

'''
def test_new_align():
    from learntools.data.tests.test_io import AWrap
    from learntools.kt.data import old_gz_to_dataset
    from learntools.data.data import align_data, align_data2

    ds = old_gz_to_dataset('data/data4.gz')
    with gzip.open('data/data5.gz', 'rb') as f:
        ds2 = Dataset.from_pickle(cPickle.load(f))
    ds2.rename_column('stim', 'skill')
    ds2.rename_column('cond', 'correct')
    #a = ds['eeg'][:20]
    #b = ds2['eeg'][:20]
    a = align_data2('data/task_data4.gz', 'data/eeg_data4.gz')
    b = align_data('data/task_data5.gz', 'data/eeg_data5.gz')
    print len(a), len(b)
    assert a[50:61] == b[50:61]

def test_new_align():
    from learntools.kt.data import old_gz_to_dataset, prepare_new_data3
    from itertools import izip
    from learntools.data.dataset import format_time
    ds = old_gz_to_dataset('data/data4.gz')
    with gzip.open('data/data5.gz', 'rb') as f:
        ds2 = Dataset.from_pickle(cPickle.load(f))
    ds2.rename_column('stim', 'skill')
    ds2.rename_column('cond', 'correct')
    print ds2.headers, ds.headers
    ds3 = prepare_new_data3(ds2)
    print ds3.headers
    diff = ds['start_time'][0] - ds3['start_time'][0]

    mask = [True] * ds2.n_rows
    mask[264] = False
    mask[298] = False
    mask[6744] = False
    mask2 = [True] * ds.n_rows
    mask2[15216] = False
    ds.mask(mask2)
    ds3.mask(mask)

    print format_time(138367661679)
    cnt = 0
    for i, (a, b0) in enumerate(izip(ds['start_time'], ds3['start_time'])):
        b = b0 + diff
        if(abs(a - b) > 10):
            print i, a - b, a, b
            cnt += 1
            if cnt > 10:
                break
    assert all(ds['correct'] == ds3['correct'])
    ds.mode = Dataset.ORIGINAL
    ds3.mode = Dataset.ORIGINAL
    assert ds['skill'][:10] == ds3['skill'][:10]

@use_logger_in_test
def test_new_align():
    from learntools.kt.data import prepare_new_data2
    a = prepare_new_data2(1, top_n=which_ds=0)
    b = prepare_new_data2(1, which_ds=1)
    assert all(a == b)

'''
