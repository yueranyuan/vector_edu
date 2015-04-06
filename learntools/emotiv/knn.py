from __future__ import division

from sklearn.neighbors import KNeighborsClassifier

from learntools.libs.logger import log_me
from learntools.emotiv.skmodel import SKModel


class KNN(SKModel):
    @log_me('...building knn')
    def __init__(self, prepared_data, n_neighbors=5, serialized=None, **kwargs):
        if serialized:
            classifier = serialized
        else:
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        super(KNN, self).__init__(prepared_data, classifier=classifier, **kwargs)