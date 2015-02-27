import os
from itertools import izip

from learntools.kt.data import (convert_task_from_xls, convert_eeg_from_xls,
                                align_data)


# create an object to stand for argument-not-set to catch inadvertent None
class NotSet:
    pass

NOTSET = NotSet

this_dir = os.path.dirname(os.path.realpath(__file__))

SAMPLE_DATA = os.path.join(this_dir, 'short_data.xls')
SAMPLE_TASK = os.path.join(this_dir, 'sample_task.gz')
SAMPLE_EEG = os.path.join(this_dir, 'sample_eeg.gz')


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
    match(subject_pairs, [('subject1', 0), ('subject2', 1)], trans_func=set)
    match(skill, [0, 1, 2, 3, 0])
    match(correct, [1, 2, 1, 2, 2])
    match(sigqual, [0, 0, 0, 200, 50])
    match(eeg, ["10 20 30 40 50", "20 30 40 50 60", "30 40 50 60 70", "10 20 30 40 50", "20 30 40 50 60"])
    true_features = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]
    if features is not NOTSET:
        for f, f2 in izip(features, true_features):
            if f2 is None:
                assert f == f2
            else:
                assert all(f) == all(f2)


def assert_aligned_data(subject=NOTSET, start_time=NOTSET, end_time=NOTSET, skill=NOTSET,
                        correct=NOTSET, subject_pairs=NOTSET, stim_pairs=NOTSET, sigqual=NOTSET,
                        eeg=NOTSET, features=NOTSET):
    match(subject, [0, 0, 1, 1])
    match(start_time, [138184295148, 138184295159, 138184414002, 138184417717])
    match(end_time, [138184295158, 138184295187, 138184414063, 138184417778])
    match(stim_pairs, [('DAD', 3), ('NEW', 1), ('IS', 2), ('THE', 0)], trans_func=set)
    match(subject_pairs, [('subject1', 0), ('subject2', 1)], trans_func=set)
    match(skill, [0, 1, 2, 0])
    match(correct, [1, 2, 1, 2])
    match(sigqual, [0, 0, 0, 50])
    match(eeg, ["10 20 30 40 50", "20 30 40 50 60", "30 40 50 60 70", "20 30 40 50 60"])
    true_features = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]
    if features is not NOTSET:
        for f, f2 in izip(features, true_features):
            if f2 is None:
                assert f == f2
            else:
                assert all(f) == all(f2)


def test_task():
    data = convert_task_from_xls(SAMPLE_DATA)
    assert_sample_data(subject=data.get_data('subject'),
                       subject_pairs=data.get_column('subject').enum_pairs,
                       start_time=data.get_data('start_time'),
                       end_time=data.get_data('end_time'),
                       skill=data.get_data('stim'),
                       stim_pairs=data.get_column('stim').enum_pairs)


def test_eeg():
    data = convert_eeg_from_xls(SAMPLE_DATA)
    assert_sample_data(subject=data.get_data('subject'),
                       subject_pairs=data.get_column('subject').enum_pairs,
                       start_time=data.get_data('start_time'),
                       end_time=data.get_data('end_time'),
                       sigqual=data.get_data('sigqual'))


def test_align():
    convert_task_from_xls(SAMPLE_DATA, SAMPLE_TASK)
    convert_eeg_from_xls(SAMPLE_DATA, SAMPLE_EEG)
    data = align_data(SAMPLE_TASK, SAMPLE_EEG)
    assert_aligned_data(subject=data.get_data('subject'),
                        subject_pairs=data.get_column('subject').enum_pairs,
                        start_time=data.get_data('start_time'),
                        correct=data.get_data('cond'),
                        skill=data.get_data('stim'),
                        stim_pairs=data.get_column('stim').enum_pairs,
                        features=data.get_data('eeg'))
