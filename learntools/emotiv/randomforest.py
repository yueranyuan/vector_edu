from __future__ import division

from sklearn.ensemble import RandomForestClassifier

from learntools.libs.logger import log_me
from learntools.emotiv.skmodel import SKModel


class RandomForest(SKModel):
    @log_me('...building RandomForest')
    def __init__(self, prepared_data, n_estimators=50, serialized=None, **kwargs):
        if serialized:
            classifier = serialized
        else:
            classifier = RandomForestClassifier(n_estimators=n_estimators)
        super(RandomForest, self).__init__(prepared_data, classifier=classifier, **kwargs)