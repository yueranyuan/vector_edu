from __future__ import division

import numpy as np
from sklearn.decomposition import PCA

from learntools.libs.logger import log
from learntools.model import Model
from learntools.libs.auc import auc


class SKModel(Model):
    def __init__(self, prepared_data, classifier, pca_components=None, pca_percentage=None, **kwargs):
        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data

        xs = ds.get_data('eeg')
        ys = ds.get_data('condition')
        self.train_x = xs[train_idx]
        self.train_y = ys[train_idx]
        self.valid_x = xs[valid_idx]
        self.valid_y = ys[valid_idx]
        self.c = classifier

        # setup PCA
        if pca_percentage:
            pca_components = self.train_x.shape[1] * pca_percentage
        if pca_components:
            self.pca = PCA(n_components=pca_components)
        else:
            self.pca = None

    def train_full(self, strategy=None, **kwargs):
        return super(SKModel, self).train_full(strategy=train_skmodel)

    def serialize(self):
        return self.c

    def predict(self, x):
        return self.c.predict(x)

    @property
    def validation_predictions(self):
        return self.predict(self.valid_x)


def train_skmodel(model, average_n_predictions=None, binarize=False, **kwargs):
    train_x = model.train_x
    valid_x = model.valid_x

    if model.pca:
        train_x = model.pca.fit_transform(train_x)
        valid_x = model.pca.transform(valid_x)

    model.c.fit(train_x, model.train_y)
    preds = model.c.predict(valid_x)
    if average_n_predictions:
        correct = 0
        incorrect = 0
        for y in np.unique(model.valid_y):
            y_preds = preds[np.where(model.valid_y == y)]
            for i in xrange(0, len(y_preds), average_n_predictions):
                if sum(y_preds[i:(i + average_n_predictions)] == y) > (average_n_predictions / 2):
                    correct += 1
                else:
                    incorrect += 1
        acc = correct / (correct + incorrect)
    else:
        acc = sum(np.equal(preds, model.valid_y)) / len(preds)

    _auc = auc(model.valid_y[:len(preds)], preds, pos_label=1)
    if binarize:
        binary_preds = np.greater_equal(preds, np.median(preds))
        acc = sum(np.equal(binary_preds, model.valid_y)) / len(binary_preds)
    print("validation auc: {auc}".format(auc=_auc))
    log('epoch 0, validation accuracy {acc:.2%}'.format(acc=acc), True)
    return acc, 0, model.serialize()