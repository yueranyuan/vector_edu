from learntools.data import Dataset
from learntools.data.crossvalidation import cv_split_within_column


def test_cv_within_basic():
    subjects = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]

    ds = Dataset(headers=(('subject', Dataset.INT), ), n_rows=len(subjects))
    for i, row in enumerate(subjects):
        ds[i] = (row,)

    train_idxs, valid_idxs = cv_split_within_column(ds, key='subject', percent=0.5, min_length=4)
    assert(train_idxs == [2, 3, 7, 8])
    assert(valid_idxs == [0, 1, 4, 5, 6])


def test_cv_within_shuffled():
    subjects = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2]

    ds = Dataset(headers=(('subject', Dataset.INT), ), n_rows=len(subjects))
    for i, row in enumerate(subjects):
        ds[i] = (row,)

    train_idxs, valid_idxs = cv_split_within_column(ds, key='subject', percent=0.5, min_length=4)
    assert(train_idxs == [6, 9, 10, 11])
    assert(valid_idxs == [0, 3, 1, 4, 7])