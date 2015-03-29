from math import ceil
from itertools import imap, groupby
from operator import or_
import random

import numpy as np

from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.libs.logger import log


def _cv_split_helper(splits, fold_index=0, percent=None):
    """
    Computes the `fold_index`th fold of `splits` of size `percent` * len(splits)
    """
    if percent is not None:
        n_heldout = int(ceil(len(splits) * percent))
        idxs = [i % len(splits) for i in xrange(fold_index * n_heldout, (fold_index + 1) * n_heldout)]
        heldout = splits[idxs]
    else:
        heldout = [splits[fold_index % len(splits)]]
    return heldout

def cv_split_randomized(ds, fold_index=0, percent=0.1, seed=0xbada55, y_column='condition', balance=True, **kwargs):
    """
    We need to generate cv splits such that the proportion of classes in different folds is the same
    and also have the subjects within each fold be up to randomness. Also, the folds should be disjoint.
    """
    rng = np.random.RandomState(seed)

    # split the data by condition
    condition_idxs = {}
    n_conditions = len(ds[y_column].enum_pairs)
    for _, condition in ds[y_column].ienum_pairs:
        mask = ds[y_column] == condition
        condition_idxs[condition] = mask_to_idx(mask)

    if balance:
        min_condition_n = min(len(idxs) for idxs in condition_idxs.values())
        for key, idxs in condition_idxs.iteritems():
            condition_idxs[key] = np.asarray(random.sample(idxs, min_condition_n))

    # generate the random disjoint folds, keyed by condition
    train_condition_idxs = {}
    valid_condition_idxs = {}
    for condition, condition_idx in condition_idxs.items():
        # we index into condition_idx so we can use logical_not to retrieve train indices from valid indices
        idx_idxs = np.arange(len(condition_idx))
        shuffled_idx_idxs = rng.permutation(idx_idxs)
        valid_idx_idxs = _cv_split_helper(shuffled_idx_idxs, fold_index=fold_index, percent=percent)
        valid_condition_idxs[condition] = condition_idx[valid_idx_idxs]
        train_idx_idxs = mask_to_idx(np.logical_not(idx_to_mask(valid_idx_idxs, mask_len=len(condition_idx))))
        train_condition_idxs[condition] = condition_idx[train_idx_idxs]

    # collect the conditions together
    train_idx = np.sort(np.concatenate(train_condition_idxs.values()))
    valid_idx = np.sort(np.concatenate(valid_condition_idxs.values()))

    # log it
    train_class_n = [sum(ds['condition'][train_idx] == cond) for cond in xrange(n_conditions)]
    valid_class_n = [sum(ds['condition'][valid_idx] == cond) for cond in xrange(n_conditions)]
    all_class_n = np.array(train_class_n) + np.array(valid_class_n)
    log("classes sizes: {}".format(all_class_n), True)
    log("training classes sizes: {}".format(train_class_n), True)
    log("validation classes sizes: {}".format(valid_class_n), True)
    log("index {} are held out".format(valid_idx), True)

    return train_idx, valid_idx


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