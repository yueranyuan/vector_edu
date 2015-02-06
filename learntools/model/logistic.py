__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

from learntools.model.net import NetworkComponent


class LogisticRegression(NetworkComponent):
    def __init__(self, n_in, n_out, activation=T.nnet.softmax, name='logistic'):
        super(LogisticRegression, self).__init__(name=name)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name=self.subname('W'),
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name=self.subname('b'),
            borrow=True
        )
        self.activation = activation

        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()

    def instance(self, x, **kwargs):
        lin_output = T.dot(x, self.W) + self.b
        lin_output.name = self.subname('lin_output')
        return self.activation(lin_output)
