from itertools import izip

from learntools.libs.data.data import (convert_task_from_xls, convert_eeg_from_xls,
                                       align_data)


# create an object to stand for argument-not-set to catch inadvertent None
class NotSet:
    pass

NOTSET = NotSet

SAMPLE_DATA = 'learntools/libs/data/tests/sample_data.xls'
SAMPLE_TASK = 'learntools/libs/data/tests/sample_task.gz'
SAMPLE_EEG = 'learntools/libs/data/tests/sample_eeg.gz'


def match(o, e, trans_func=None):
    if o is NOTSET:
        return
    if trans_func:
        o = trans_func(o)
        e = trans_func(e)
    try:
        assert o == e
    except ValueError:
        assert all(o == e)


def assert_sample_data(subject=NOTSET, start_time=NOTSET, end_time=NOTSET, skill=NOTSET,
                       correct=NOTSET, subject_pairs=NOTSET, stim_pairs=NOTSET, sigqual=NOTSET,
                       eeg=NOTSET, features=NOTSET):
    match(subject, [0, 0, 1, 1, 1])
    match(start_time, [138184295148, 138184295159, 138184414002, 138184414064, 138184417717])
    match(end_time, [138184295158, 138184295187, 138184414063, 138184414129, 138184417778])
    match(stim_pairs, [('DAD', 3), ('NEW', 1), ('IS', 2), ('THE', 0)], trans_func=set)
    match(subject_pairs, [('fAH6-6-2004-06-18', 0), ('fAJ7-7-2007-02-07', 1)], trans_func=set)
    match(skill, [0, 1, 2, 3, 0])
    match(correct, [1, 2, 1, 2, 2])
    match(sigqual, [0, 0, 0, 200, 50])
    match(eeg, ["10 20 30 40 50", "20 30 40 50 60", "30 40 50 60 70", "10 20 30 40 50", "20 30 40 50 60"])
    true_features = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], None, [0., 0., 0., 0.]]
    if features is not NOTSET:
        for f, f2 in izip(features, true_features):
            if f2 is None:
                assert f == f2
            else:
                assert all(f) == all(f2)


def test_task():
    data = convert_task_from_xls(SAMPLE_DATA)
    assert_sample_data(*data)


def test_eeg():
    subject, start_time, end_time, sigqual, eeg_freq, subject_pairs = (
        convert_eeg_from_xls(SAMPLE_DATA))
    assert_sample_data(subject=subject, start_time=start_time, end_time=end_time,
                       sigqual=sigqual, subject_pairs=subject_pairs)


def test_align():
    convert_task_from_xls(SAMPLE_DATA, SAMPLE_TASK)
    convert_eeg_from_xls(SAMPLE_DATA, SAMPLE_EEG)
    subject, skill, correct, task_start, features, stim_pairs = align_data(SAMPLE_TASK, SAMPLE_EEG)
    assert_sample_data(subject=subject, skill=skill, correct=correct, start_time=task_start,
                       stim_pairs=stim_pairs, features=features)
