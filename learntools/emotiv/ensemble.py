from __future__ import division

import cPickle as pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from learntools.libs.logger import log_me, log
from learntools.emotiv.skmodel import SKModel
from learntools.emotiv.randomforest import RandomForest
from learntools.model.model import Model


class ModelType(object):
    BASE = 0
    RAW_BASE = 1
    SUBJECT = 2
    AUTOENCODER = 3
    BATCH_NORM = 4
    SVM = 5
    MULTISTAGE_BATCH_NORM = 6
    RANDOMFOREST = 7
    ENSEMBLE = 8


def load_classifier(prepared_data, classifier_file, **kwargs):
    with open(classifier_file, 'r') as f:
        content = pickle.load(f)
    model_type = content['model_type']
    serialized = content['params']
    if model_type == ModelType.RANDOMFOREST:
        classifier = RandomForest(prepared_data, serialized=serialized, **kwargs)
    else:
        raise Exception("invalid model type")
    return classifier


class Ensemble(Model):
    @log_me("... building Ensemble")
    def __init__(self, prepared_data, saved_classifiers=None, **kwargs):
        if not saved_classifiers:
            raise Exception("no classifiers loaded")
        self.classifiers = [load_classifier(prepared_data, classifier_file, **kwargs)
                            for classifier_file in saved_classifiers]

        ds, train_idx, valid_idx = prepared_data

        xs = ds.get_data('eeg')
        ys = ds.get_data('condition')
        self.train_x = xs[train_idx]
        self.train_y = ys[train_idx]
        self.valid_x = xs[valid_idx]
        self.valid_y = ys[valid_idx]

    def train_full(self, strategy=None, **kwargs):
        return super(Ensemble, self).train_full(strategy=train_ensemble)


def train_ensemble(model, **kwargs):
    all_preds = np.asarray([c.predict(model.valid_x) for c in model.classifiers])
    preds = np.sum(all_preds, axis=0)
    binary_preds = np.greater_equal(preds, np.median(preds))
    acc = sum(np.equal(binary_preds, model.valid_y)) / len(binary_preds)
    log('epoch 0, validation accuracy {acc:.2%}'.format(acc=acc), True)
    return acc, 0, None


class LogRegEnsemble(SKModel):
    @log_me('...building Ensemble')
    def __init__(self, prepared_data, serialized=None, **kwargs):
        if serialized:
            raise Exception("ensemble model is unserializable")
        else:
            classifier = _LogRegEnsemble()
        super(LogRegEnsemble, self).__init__(prepared_data, classifier=classifier, **kwargs)


class _LogRegEnsemble():
    def __init__(self, n_channels=14, n_features=4):
        self.n_channels = n_channels
        self.n_features = n_features

    def fit(self, x, y):
        models = []
        preds = np.zeros((len(x), self.n_channels + self.n_features))

        # create channel based models
        for i in xrange(self.n_channels):
            print('training channel model {}'.format(i))
            model = LogisticRegression()
            feats = x[:, (i * self.n_features):((i + 1) * self.n_features)]
            model.fit(feats, y)
            models.append(model)
            preds[:, i] = model.predict(feats)

        # create band based models
        for i in xrange(self.n_features):
            print('training band model {}'.format(i))
            model = LogisticRegression()
            feats = x[:, i:(self.n_channels * self.n_features):self.n_features]
            model.fit(feats, y)
            models.append(model)
            preds[:, self.n_channels + i] = model.predict(feats)

        # create integrating forest
        top_classifier = RandomForestClassifier()
        top_classifier.fit(preds, y)

        self.models = models
        self.c = top_classifier

    def predict(self, x):
        preds = np.zeros((len(x), self.n_channels + self.n_features))

        for i in xrange(self.n_channels):
            model = self.models[i]
            feats = x[:, (i * self.n_features):((i + 1) * self.n_features)]
            preds[:, i] = model.predict(feats)

        # create band based models
        for i in xrange(self.n_features):
            model = self.models[self.n_channels + i]
            feats = x[:, i:(self.n_channels * self.n_features):self.n_features]
            preds[:, self.n_channels + i] = model.predict(feats)

        return self.c.predict(preds)