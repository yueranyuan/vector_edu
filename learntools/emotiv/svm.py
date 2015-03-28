from __future__ import division

import numpy as np
from sklearn import svm

from learntools.libs.logger import log_me, log
from learntools.model import Model


class SVM(Model):
    @log_me('...building SVM')
    def __init__(self, prepared_data, **kwargs):
        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data

        xs = ds.get_data('eeg')
        ys = ds.get_data('condition')
        self.train_x = xs[train_idx]
        self.train_y = ys[train_idx]
        self.valid_x = xs[valid_idx]
        self.valid_y = ys[valid_idx]

        self.svm = svm.SVC(kernel='poly', degree=2, C=0.001)

    def train_full(self, **kwargs):
        self.svm.fit(self.train_x, self.train_y)
        preds = self.svm.predict(self.valid_x)
        log("final accuracy: {}".format(sum(np.equal(preds, self.valid_y)) / len(preds)), True)