from __future__ import division

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from learntools.libs.logger import log_me
from learntools.emotiv.skmodel import SKModel


class Ensemble(SKModel):
    @log_me('...building Ensemble')
    def __init__(self, prepared_data, **kwargs):
        super(Ensemble, self).__init__(prepared_data, **kwargs)
        self.c = LogRegEnsemble()


class LogRegEnsemble():
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