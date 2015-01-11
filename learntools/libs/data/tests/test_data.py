from learntools.libs.data.data import convert_task_from_xls


def test_module():
    data = convert_task_from_xls('learntools/libs/data/tests/sample_data.xls')
    subject, start_time, end_time, skill, correct, subject_pairs, stim_pairs = data

    assert all(subject == [0, 0, 1, 1, 1])
    assert all(start_time == [138184295148, 138184295159, 138184414002, 138184414064, 138184417717])
    assert all(end_time == [138184295158, 138184295187, 138184414063, 138184414129, 138184417778])
    assert set(stim_pairs) == set([('DAD', 3), ('NEW', 1), ('IS', 2), ('THE', 0)])
    assert set(subject_pairs) == set([('fAH6-6-2004-06-18', 0), ('fAJ7-7-2007-02-07', 1)])
    assert all(skill == [0, 1, 2, 3, 0])
    assert all(correct == [1, 2, 1, 2, 2])
