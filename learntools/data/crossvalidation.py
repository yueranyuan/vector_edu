from math import ceil
from itertools import imap, groupby
from operator import or_

import numpy as np

from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.libs.logger import log


def _cv_split_helper(splits, fold_index=0, percent=None):
    if percent is not None:
        n_heldout = int(ceil(len(splits) * percent))
        idxs = [i % len(splits) for i in xrange(fold_index * n_heldout, (fold_index + 1) * n_heldout)]
        heldout = splits[idxs]
    else:
        heldout = [splits[fold_index % len(splits)]]
    return heldout


def cv_split(ds, fold_index=0, split_on=None, percent=None, **kwargs):
    # cross-validation split
    if split_on:
        splits = np.unique(ds[split_on])
        heldout = _cv_split_helper(splits, fold_index=fold_index, percent=percent)

        mask = reduce(or_, imap(lambda s: np.equal(ds[split_on], s), heldout))
        train_idx = np.nonzero(np.logical_not(mask))[0]
        valid_idx = np.nonzero(mask)[0]
    else:
        heldout = _cv_split_helper(np.asarray(range(ds.n_rows)), fold_index=fold_index, percent=percent)
        valid_idx = heldout
        train_mask = np.logical_not(idx_to_mask(valid_idx, mask_len=ds.n_rows))
        train_idx = mask_to_idx(train_mask)

    # print/log what we held out
    split_on_str = split_on if split_on else 'index'
    info = '{split_on} {heldout} are held out'.format(split_on=split_on_str, heldout=heldout)
    try:
        log(info, True)
    except:
        print '[was not logged] {}'.format(info)

    return train_idx, valid_idx


def cv_split_within_column(ds, fold_index=0, key=None, percent=None, min_length=None, **kwargs):
    idx_sorted_by_key = sorted(range(ds.n_rows), key=lambda i: ds[key][i])
    valid_idx = []
    train_idx = []

    for _, igroup in groupby(idx_sorted_by_key, lambda i: ds[key][i]):
        group = list(igroup)
        # skip key groups that have less than a minimal size
        if min_length and len(group) < min_length:
            continue
        # hold out a certain portion of the group via fold_index
        heldout = _cv_split_helper(group, fold_index=fold_index, percent=percent)
        valid_idx.extend(heldout)
        _train_idx = [g for g in group if g not in heldout]
        train_idx.extend(_train_idx)
    return train_idx, valid_idx