from __future__ import division

from itertools import compress
from collections import Counter

import numpy as np

from learntools.libs.auc import auc
from learntools.kt.base import BaseKT

from theano import config

config.exception_verbosity = 'high'


class MajorityKT(BaseKT):
    '''a trainable, applyable model for neighborhood based kt with a previous result as reference
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    def __init__(self, prepared_data, skill_matrix,
                 **kwargs):
        '''
        Args:
            prepared_data (tuple(Dataset, int[], int[])): a tuple that holds the data to be used,
                the row indices of the training set, and the row indices of the validation set
        '''
        # ##########
        # STEP1: order the data properly so that we can read from it sequentially
        # when training the model

        ds, train_idx, valid_idx = prepared_data

        self.skill_x = ds.get_data('skill')
        self.correct_y = ds.get_data('correct') - 1
        skill_count = Counter(self.skill_x[train_idx])
        correct_skill_count = Counter(compress(self.skill_x[train_idx], self.correct_y[train_idx]))

        self.skill_average = {skill: correct_skill_count[skill] / cnt for skill, cnt in skill_count.iteritems()}
        self.percent_correct = np.average(self.correct_y)
        self.valid_idx = valid_idx

    def train_full(self, **kwargs):
        print('predicting...')
        preds = [self.skill_average.get(skill, self.percent_correct) for skill in self.skill_x[self.valid_idx]]
        test_y = self.correct_y[self.valid_idx]

        print(auc(test_y, preds, pos_label=1))