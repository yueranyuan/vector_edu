__docformat__ = 'restructedtext en'

import theano.tensor as T

from learntools.model.net import NetworkComponent, HiddenNetwork, HiddenLayer


class MLP(NetworkComponent):
    def __init__(self, n_in, size, n_out, activation='rectifier', output_activation='softmax',
                 name='MLP', rng_state=None):
        super(MLP, self).__init__(name=name)
        self.hidden = HiddenNetwork(
            rng_state=rng_state,
            size=[n_in] + size,
            activation=activation,
            name=self.subname('hiddenlayers')
        )

        log_layer_n_in = size[-1] if len(size) else n_in
        self.logRegressionLayer = HiddenLayer(
            n_in=log_layer_n_in,
            n_out=n_out,
            activation=output_activation,
            name=self.subname('outputlayer')
        )
        self.components = [self.hidden, self.logRegressionLayer]

    def instance(self, x, dropout=0, **kwargs):
        x1 = self.hidden.instance(x, dropout=dropout, **kwargs)
        return self.logRegressionLayer.instance(x1, **kwargs)

    def output(self, pY):
        return pY[:, 1] - pY[:, 2]

    def negative_log_likelihood(self, pY, y):
        return -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
