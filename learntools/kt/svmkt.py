from sklearn import svm

from learntools.libs.logger import log_me
from learntools.kt.pairkt import PairKT

from theano import config

config.exception_verbosity = 'high'


class SvmKT(PairKT):
    '''a trainable, applyable model for logistic regression based kt with a previous result as reference
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    @log_me('...building svmkt')
    def __init__(self, prepared_data, skill_matrix,
                **kwargs):
        self.classifier = svm.SVR(kernel='poly', degree=2, C=0.001)
        super(SvmKT, self).__init__(prepared_data, skill_matrix, **kwargs)