import numpy as np
import theano
import theano.tensor as T

from model.mlp import MLP, rectifier


def test_mlp_smoke():
    rng = np.random.RandomState(1234)
    net = MLP(rng, 2, [10], 2, activation=rectifier, dropout=None)
    x = T.dmatrix()
    y = net.instance(x, rng=rng)
    grads = T.grad(T.sum(y), net.params)

    f = theano.function(inputs=[x, net.dropout], outputs=[y] + grads)
    outs = f([[1, 1], [2, 2]], 0.1)

    for i, o in enumerate(outs):
        assert not any(np.isnan(o).flatten())
