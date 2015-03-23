from __future__ import division

from math import sqrt
from itertools import groupby, compress
import itertools
from collections import defaultdict, Counter

import numpy as np

from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.libs.auc import auc
from learntools.kt.base import BaseKT

from theano import config

config.exception_verbosity = 'high'


class NeighborKT(BaseKT):
    '''a trainable, applyable model for neighborhood based kt with a previous result as reference
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    def __init__(self, prepared_data, skill_matrix,
                 current_eeg_on=1, **kwargs):
        '''
        Args:
            prepared_data (tuple(Dataset, int[], int[])): a tuple that holds the data to be used,
                the row indices of the training set, and the row indices of the validation set
        '''
        # ##########
        # STEP1: order the data properly so that we can read from it sequentially
        # when training the model

        ds, train_idx, valid_idx = prepared_data
        N = len(ds['correct'])
        eeg_vector_len = ds.get_data('eeg').shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['subject']))
        valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

        sorted_i = sorted(xrange(N), key=lambda(i): (ds['subject'][i]))
        ds.reorder(sorted_i)

        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        self.train_idx = mask_to_idx(train_mask)
        self.valid_idx = mask_to_idx(valid_mask)

        self.skill_x = ds.get_data('skill')
        self.subject_x = ds.get_data('subject')
        time_x = ds.get_data('start_time')
        self.correct_y = ds.get_data('correct') - 1
        eeg_full = ds.get_data('eeg')

        # ########
        # STEP2: structure data for use in computing similarity

        # get the percentage correct
        skill_count = Counter(self.skill_x[train_idx])
        correct_skill_count = Counter(compress(self.skill_x[train_idx], self.correct_y[train_idx]))

        self.skill_average = defaultdict(itertools.repeat(np.average(self.correct_y)).next)
        for skill, cnt in skill_count.iteritems():
            self.skill_average[skill] = correct_skill_count[skill] / cnt

        # create dictionary mapping of subject to time-sorted data idxs for all training data
        self.subject_data = {subject: sorted(idxs, key=lambda(i): time_x[i])
                             for subject, idxs in groupby(xrange(N), lambda(i): self.subject_x[i])}

    def _get_nearby_tasks(self, idx, pre=50, post=50):
        subject = self.subject_x[idx]
        data = self.subject_data[subject]
        local_i = data.index(idx)
        return data[max(0, local_i - pre):min(len(data) - 1, local_i + post)]

    def _rate_pair(self, i1, i2, pre=50, post=50):
        # get nearby tasks
        nearby1 = self._get_nearby_tasks(i1, pre=pre, post=post)
        nearby2 = self._get_nearby_tasks(i2, pre=pre, post=post)

        # find overlapping skills
        skills1 = {self.skill_x[i] for i in nearby1}
        skills2 = {self.skill_x[i] for i in nearby2}
        skill_overlap = skills1 & skills2

        # take cosine similarity of overlapping skills
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        for s in skill_overlap:
            # TODO: speed this up
            s_correct = [self.correct_y[i] for i in nearby1 if self.skill_x[i] == s]
            s1_value = sum(s_correct) / len(s_correct)  # - self.skill_average[s]
            s_correct = [self.correct_y[i] for i in nearby2 if self.skill_x[i] == s]
            s2_value = sum(s_correct) / len(s_correct)  # - self.skill_average[s]
            numerator += s1_value * s2_value
            denominator1 += s1_value * s1_value
            denominator2 += s2_value * s2_value
        cosine_sim = numerator / (sqrt(denominator1) * sqrt(denominator2))

        return cosine_sim

    def _predict(self, idx, k=20, pre=50, post=50):
        print('predicting {}'.format(idx))
        skill_id = self.skill_x[idx]
        neighbors = []
        for subject, idxs in self.subject_data.iteritems():
            for i in idxs:
                if skill_id == self.skill_x[i] and i != idx:
                    rating = self._rate_pair(idx, i, pre=pre, post=post)
                    neighbors.append((rating, self.correct_y[i]))

        if len(neighbors) == 0:
            return 1.0
        neighbors = sorted(neighbors, key=lambda(rating, correct): rating)
        k_nearest = neighbors[:min(k, len(neighbors))]
        rating = sum(correct * rating for rating, correct in k_nearest) / len(k_nearest)
        return rating

    def train_full(self, k=20, pre=50, post=50, **kwargs):
        print('predicting...')
        preds = [self._predict(i, k=k, pre=pre, post=post) for i in self.valid_idx]
        test_y = [self.correct_y[i] for i in self.valid_idx]

        print(auc(test_y, preds, pos_label=1))