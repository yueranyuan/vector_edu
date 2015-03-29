from __future__ import division

import numpy as np

from learntools.libs.logger import log
from learntools.model import Model
from learntools.libs.auc import auc


class SKModel(Model):
    def __init__(self, prepared_data, classifier, **kwargs):
        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data

        xs = ds.get_data('eeg')
        ys = ds.get_data('condition')
        self.train_x = xs[train_idx]
        self.train_y = ys[train_idx]
        self.valid_x = xs[valid_idx]
        self.valid_y = ys[valid_idx]
        self.c = classifier

    def train_full(self, strategy=None, **kwargs):
        return super(SKModel, self).train_full(strategy=train_skmodel)

    def serialize(self):
        return self.c

    def predict(self, x):
        return self.c.predict(x)

    @property
    def validation_predictions(self):
        return self.predict(self.valid_x)


def train_skmodel(model, **kwargs):
    model.c.fit(model.train_x, model.train_y)
    preds = model.c.predict(model.valid_x)
    binary_preds = np.greater_equal(preds, np.median(preds))
    acc = sum(np.equal(binary_preds, model.valid_y)) / len(binary_preds)
    _auc = auc(model.valid_y[:len(preds)], preds, pos_label=1)
    print("validation auc: {auc}".format(auc=_auc))
    log('epoch 0, validation accuracy {acc:.2%}'.format(acc=acc), True)
    return acc, 0, model.serialize()