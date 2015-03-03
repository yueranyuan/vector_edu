import numpy as np
import theano
import theano.tensor as T

from learntools.model.net import BatchNormLayer, AutoencodingBatchNormLayer
from learntools.libs.common_test_utils import use_logger_in_test


def gen_data(n_in, n_x1, n_x2):
    # gen data
    x1 = [1 if i % 2 == 0 else 0 for i in xrange(n_in)]
    x2 = [1 if i % 2 == 1 else 0 for i in xrange(n_in)]
    xs = np.asarray([x1] * n_x1 + [x2] * n_x2)
    ys = np.asarray([0] * n_x1 + [1] * n_x2)

    # gen cross validation indices
    train_idx = range(0, len(xs), 2)
    valid_idx = range(1, len(xs), 2)

    return xs, ys, train_idx, valid_idx


@use_logger_in_test
def test_batchnorm():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    layer = AutoencodingBatchNormLayer(n_in=n_in, n_out=2)
    inp = T.dmatrix('xs')
    out = layer.instance(inp, inp)
    f = theano.function([inp], out[:2])
    ys = f(xs)
    assert [y.shape for y in ys] == [(800, 2), (800, 2)]


@use_logger_in_test
def test_batchnorm_serialize():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    layer = BatchNormLayer(n_in=n_in, n_out=2)
    params = layer.serialize()
    layer2 = BatchNormLayer.deserialize(parameters=params)
    np.testing.assert_array_equal(layer.t_W.get_value(borrow=True),
                                  layer2.t_W.get_value(borrow=True))
    inp = T.dmatrix('xs')
    out1 = layer.instance(inp, inp)[:2]
    out2 = layer.instance(inp, inp)[:2]
    f1 = theano.function([inp], out1)
    f2 = theano.function([inp], out2)
    ey = f1(xs)
    ey2 = f2(xs)
    np.testing.assert_array_equal(ey, ey2)


@use_logger_in_test
def test_autoencodingbatchnorm():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    layer = AutoencodingBatchNormLayer(n_in=n_in, n_out=2)
    inp = T.dmatrix('xs')
    out = layer.instance(inp, inp)
    f = theano.function([inp], out[:2] + out[3:])
    ys = f(xs)
    print ys[-1]
    assert [y.shape for y in ys] == [(800, 2), (800, 2), (800, 100), (800, 100)]


@use_logger_in_test
def test_autoencodingbatchnorm_serialize():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    layer = AutoencodingBatchNormLayer(n_in=n_in, n_out=2)
    params = layer.serialize()
    layer2 = AutoencodingBatchNormLayer.deserialize(parameters=params)
    np.testing.assert_array_equal(layer.t_W.get_value(borrow=True),
                                  layer2.t_W.get_value(borrow=True))
    inp = T.dmatrix('xs')
    out1 = layer.instance(inp, inp)[0]
    out2 = layer.instance(inp, inp)[0]
    f1 = theano.function([inp], out1)
    f2 = theano.function([inp], out2)
    ey = f1(xs)
    ey2 = f2(xs)
    np.testing.assert_array_equal(ey, ey2)