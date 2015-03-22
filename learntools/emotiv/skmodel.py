from __future__ import division

import numpy as np

from learntools.libs.logger import log
from learntools.model import Model


class SKModel(Model):
    def __init__(self, prepared_data, n_estimators=50, **kwargs):
        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data

        xs = ds.get_data('eeg')
        ys = ds.get_data('condition')
        self.train_x = xs[train_idx]
        self.train_y = ys[train_idx]
        self.valid_x = xs[valid_idx]
        self.valid_y = ys[valid_idx]

    def train_full(self, strategy=None, **kwargs):
        return super(SKModel, self).train_full(strategy=train_skmodel)


def train_skmodel(model, **kwargs):
    model.c.fit(model.train_x, model.train_y)
    preds = model.c.predict(model.valid_x)
    acc = sum(np.equal(preds, model.valid_y)) / len(preds)
    log('epoch 0, validation accuracy {acc:.2%}'.format(acc=acc), True)
    return acc, 0, None