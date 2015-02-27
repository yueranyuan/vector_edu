__docformat__ = 'restructedtext en'

import theano.tensor as T

from learntools.model.net import HiddenLayer


class LogisticRegression(HiddenLayer):
    def instance(self, x, **kwargs):
        return T.nnet.softmax(T.dot(x, self.t_W) + self.t_b)
