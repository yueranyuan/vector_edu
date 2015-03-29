from __future__ import division

from sklearn import svm

from learntools.libs.logger import log_me
from learntools.emotiv.skmodel import SKModel


class SVM(SKModel):
    @log_me('...building SVM')
    def __init__(self, prepared_data, **kwargs):
        super(SVM, self).__init__(prepared_data, **kwargs)
        self.c = svm.SVC(kernel='poly', degree=2, C=0.001)