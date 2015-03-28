__docformat__ = 'restructedtext en'

import theano.tensor as T

from learntools.model.logistic import LogisticRegression
from learntools.model.math import rectifier
from learntools.model.net import NetworkComponent, HiddenNetwork, ConvolutionalNetwork


class MLP(NetworkComponent):
    def __init__(self, rng, n_in, size, n_out, activation=rectifier, output_activation=T.nnet.softmax,
                 dropout=None, name='MLP'):
        super(MLP, self).__init__(name=name)
        self.dropout = T.scalar('dropout') if dropout is None else dropout
        self.hidden = HiddenNetwork(
            rng=rng,
            n_in=n_in,
            size=size,
            activation=activation,
            dropout=self.dropout,
            name=self.subname('hiddenlayer')
        )

        self.logRegressionLayer = LogisticRegression(
            n_in=self.hidden.n_out,
            n_out=n_out,
            activation=output_activation,
            name=self.subname('outputlayer')
        )
        self.components = [self.hidden, self.logRegressionLayer]

    def instance(self, x, **kwargs):
        x1 = self.hidden.instance(x, **kwargs)
        return self.logRegressionLayer.instance(x1, **kwargs)

    def output(self, pY):
        return pY[:, 1] - pY[:, 2]

    def negative_log_likelihood(self, pY, y):
        return -T.mean(T.log(pY)[T.arange(y.shape[0]), y])


class ConvolutionalMLP(NetworkComponent):
    def __init__(self, rng, n_in, size, n_out, activation=rectifier,
                 dropout=None, field_width=3, ds_factor=2, name='MLP'):
        super(ConvolutionalMLP, self).__init__(name=name)
        self.dropout = T.scalar('dropout') if dropout is None else dropout
        self.hidden = ConvolutionalNetwork(
            rng=rng,
            n_in=n_in,
            size=size,
            activation=activation,
            dropout=self.dropout,
            field_width=field_width,
            ds_factor=ds_factor,
            name=self.subname('convolutional')
        )

        conv_n_out = self.hidden.n_out
        self.logRegressionLayer = LogisticRegression(
            n_in=conv_n_out,
            n_out=n_out,
            name=self.subname('softmax')
        )
        self.components = [self.hidden, self.logRegressionLayer]

    def instance(self, x, **kwargs):
        x1 = self.hidden.instance(x, **kwargs)
        return self.logRegressionLayer.instance(x1, **kwargs)

    def output(self, pY):
        return pY[:, 1] - pY[:, 2]

    def negative_log_likelihood(self, pY, y):
        return -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
