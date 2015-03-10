import numpy as np
import theano
import theano.tensor as T

from learntools.model.tests.test_autoencoder import gen_data
from learntools.model.net import BatchNormLayer, AutoencodingBatchNormLayer, DecodingBatchNormLayer
from learntools.libs.common_test_utils import use_logger_in_test


@use_logger_in_test
def test_batchnorm():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    layer = BatchNormLayer(n_in=n_in, n_out=2)
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


@use_logger_in_test
def test_batchnorm_decode():
    n_in, n_out = 100, 2
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    autoencoder = AutoencodingBatchNormLayer(n_in=n_in, n_out=n_out)
    p = autoencoder.serialize()
    inp = T.dmatrix('xs')
    out1 = autoencoder.instance(inp, inp)
    f_encode1 = theano.function([inp], out1[:2])
    f_decode1 = theano.function([inp], out1[3:])

    encoder = BatchNormLayer(name="encoder", W=p["W"], b=p["b"], beta=p["beta"], gamma=p["gamma"], mean=p["mean"],
                             variance=p["variance"])
    decoder = DecodingBatchNormLayer(name="decoder", encoder_W=encoder.t_W, b=p["decode_b"], beta=p["decode_beta"], gamma=p["decode_gamma"],
                                     mean=p["decode_mean"], variance=p["decode_variance"])
    np.testing.assert_array_equal(autoencoder.decode_beta.get_value(borrow=True),
                                  decoder.beta.get_value(borrow=True))
    np.testing.assert_array_equal(autoencoder.decode_mean.get_value(borrow=True),
                                  decoder.mean.get_value(borrow=True))
    np.testing.assert_array_equal(autoencoder.decode_gamma.get_value(borrow=True),
                                  decoder.gamma.get_value(borrow=True))
    np.testing.assert_array_equal(autoencoder.t_decode_b.get_value(borrow=True),
                                  decoder.b)
    train_out, infer_out, _ = encoder.instance(inp, inp)
    train_recon, infer_recon, _ = decoder.instance(train_out, infer_out)
    f_encode2 = theano.function([inp], [train_out, infer_out])
    f_decode2 = theano.function([inp], [train_recon, infer_recon])

    y_encode_1 = f_encode1(xs)
    y_decode_1 = f_decode1(xs)
    y_encode_2 = f_encode2(xs)
    y_decode_2 = f_decode2(xs)

    np.testing.assert_array_equal(y_encode_1, y_encode_2)
    np.testing.assert_array_equal(y_decode_1, y_decode_2)
