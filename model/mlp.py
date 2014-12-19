__docformat__ = 'restructedtext en'


import numpy

import theano
import theano.tensor as T

from model.logistic import LogisticRegression


def rectifier(x):
    return T.maximum(x, 0)


# inspired by https://github.com/mdenil/dropout/blob/master/mlp.py
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out=None, W=None, b=None,
                 activation=rectifier, dropout=None):
        self.dropout = T.scalar('dropout') if dropout is None else dropout
        # dropouts on input
        srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
        mask = srng.binomial(n=1, p=1 - self.dropout, size=input.shape)
        # cast because int * float32 = float64 which does not run on GPU
        self.input = input * T.cast(mask, theano.config.floatX)

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = (T.dot(input, self.W) + self.b) * (1 / (1 - self.dropout))
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, activation=rectifier,
                 dropout=None):
        self.dropout = T.scalar('dropout') if dropout is None else dropout
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation,
            dropout=self.dropout
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = self.hiddenLayer.L1 + self.logRegressionLayer.L1
        self.L2_sqr = self.hiddenLayer.L2_sqr + self.logRegressionLayer.L2_sqr

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.output = self.logRegressionLayer.p_y_given_x[:, 1] - self.logRegressionLayer.p_y_given_x[:, 2]
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
