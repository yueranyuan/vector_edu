from itertools import izip, islice

import numpy as np

from learntools.libs.data.io import Column, TimeColumn, EnumColumn, MatColumn, Dataset
from test_data import assert_sample_data


strtimes = ["2013-10-15 09:15:51.480000", "2013-10-15 09:15:55.480000", "2013-10-15 09:15:55.480000"]
timestamps = [138184295148, 138184295548, 138184295548]
enumstr = ['jan', 'feb', 'jan']
enumint = [0, 1, 0]
numstr = ['0', '1', '2']
nums = [0, 1, 2]


def test_column():
    col = Column('example_column')
    col[0] = numstr[0]
    assert col[0] == nums[0]
    col[1:3] = numstr[1:3]
    assert all(col[:3] == nums)

    col2 = Column('example', data=nums)
    assert all(col2[:3] == nums)


def test_timecolumn():
    col = TimeColumn('example_column')
    strtimes = ["2013-10-15 09:15:51.480000", "2013-10-15 09:15:55.480000", "2013-10-15 09:15:55.480000"]
    col[0] = strtimes[0]
    col[1:3] = strtimes[1:3]
    assert all(col[:3] == timestamps)
    col.mode = TimeColumn.ORIGINAL
    assert col[:3] == strtimes

    col2 = TimeColumn('example', data=timestamps)
    assert all(col2[:3] == timestamps)
    col2.mode = TimeColumn.ORIGINAL
    assert col2[:3] == strtimes


def test_enumcolumn():
    col = EnumColumn('example_column')
    enumstr = ['jan', 'feb', 'jan']
    enumint = [0, 1, 0]
    col[0] = enumstr[0]
    col[1:3] = enumstr[1:3]
    assert all(col[:3] == enumint)
    col.mode = EnumColumn.ORIGINAL
    assert col[:3] == enumstr

    col2 = EnumColumn('example', data=enumint, enum_dict={'jan': 0, 'feb': 1})
    assert all(col2[:3] == enumint)
    col2.mode = EnumColumn.ORIGINAL
    assert col2[:3] == enumstr


def test_matcolumn():
    col = MatColumn('example_column')
    matstr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    col[0] = matstr[0]
    assert all(col[0] == matstr[0])
    col[1:3] = matstr[1:3]
    assert np.all(col[:3] == matstr)


def test_dataset():
    dataset = Dataset([('int', Dataset.INT), ('enum', Dataset.ENUM), ('time', Dataset.TIME)], n_rows=len(nums))
    for i, row in enumerate(izip(numstr, enumstr, strtimes)):
        dataset[i] = row

    for d, row in izip(islice(dataset, None, 3), izip(nums, enumint, timestamps)):
        assert tuple(d) == row
    dataset.mode = Dataset.ORIGINAL
    for d, row in izip(islice(dataset, None, 3), izip(numstr, enumstr, strtimes)):
        assert tuple(map(str, d)) == row

'''
def test_loader():
    from learntools.libs.data.io import load
    headers = (('cond', Dataset.INT),
               ('subject', Dataset.ENUM),
               ('stim', Dataset.ENUM),
               ('block', Dataset.ENUM),
               ('start_time', Dataset.TIME),
               ('end_time', Dataset.TIME),
               ('rawwave', Dataset.STR))
    data = load('learntools/libs/data/tests/sample_data.xls', headers)

    subject = data.get_column('subject')
    correct = data.get_column('cond')
    skill = data.get_column('stim')
    start_time = data.get_column('start_time')
    end_time = data.get_column('end_time')
    stim_pairs = data.get_column('stim').enum_pairs
    subject_pairs = data.get_column('subject').enum_pairs

    assert_sample_data(subject, start_time, end_time, skill, correct, subject_pairs, stim_pairs)

    eeg = data.get_column("rawwave")
    assert_sample_data(eeg=eeg)


def test_pickle():
    dataset = Dataset([('int', Dataset.INT), ('enum', Dataset.ENUM), ('time', Dataset.TIME)], n_rows=len(nums))
    for i, row in enumerate(izip(numstr, enumstr, strtimes)):
        dataset[i] = row

    import cPickle
    gz_name = 'learntools/libs/data/tests/sample_data.gz'
    with open(gz_name, 'w') as f:
        cPickle.dump(dataset.to_pickle(), f)

    with open(gz_name, 'r') as f:
        dataset2 = Dataset.from_pickle(cPickle.load(f))

    for d, d2 in izip(dataset, dataset2):
        assert d == d2
'''
