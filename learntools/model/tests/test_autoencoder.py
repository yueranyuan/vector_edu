import numpy as np

from learntools.model.autoencoder import Autoencoder, MLP
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
def test_autoencoder_fit():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    size = [n_in, n_in / 2]
    autoencoder = Autoencoder(size=size, weights=None)
    autoencoder_score = autoencoder.fit(xs, train_idx, valid_idx, n_epochs=100)
    assert autoencoder_score < 0.0001


@use_logger_in_test
def test_autoencoder_reconstruct():
    n_in = 100
    xs, _, _, _ = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    size = [n_in, n_in / 2]
    autoencoder = Autoencoder(size=size, weights=None)
    ey = autoencoder.reconstruct(xs)
    assert ey.shape == xs.shape
    assert 0 < np.sum(ey) < ey.shape[0] * ey.shape[1]


@use_logger_in_test
def test_autoencoder_encode():
    n_in = 100
    hidden_width = n_in / 2
    xs, _, _, _ = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    size = [n_in, hidden_width]
    autoencoder = Autoencoder(size=size, weights=None)
    ey = autoencoder.encode(xs)
    np.testing.assert_array_equal(ey.shape, (xs.shape[0], hidden_width))
    assert 0 < np.sum(ey) < ey.shape[0] * ey.shape[1]


@use_logger_in_test
def test_autoencoder_serialize():
    n_in = 100
    xs, _, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    size = [n_in, n_in / 2]
    autoencoder = Autoencoder(size=size, weights=None)
    autoencoder.fit(xs, train_idx, valid_idx, n_epochs=10)
    autoencoder_params = autoencoder.serialize()
    autoencoder2 = Autoencoder.deserialize(parameters=autoencoder_params)
    np.testing.assert_array_equal(autoencoder.encoder.layers[0].t_W.get_value(borrow=True),
                                  autoencoder2.encoder.layers[0].t_W.get_value(borrow=True))
    np.testing.assert_array_equal(autoencoder.decoder.t_W.owner.inputs[0].get_value(borrow=True),
                                  autoencoder2.decoder.t_W.owner.inputs[0].get_value(borrow=True))
    ey = autoencoder.reconstruct(xs)
    ey2 = autoencoder2.reconstruct(xs)
    np.testing.assert_array_equal(ey, ey2)


@use_logger_in_test
def test_classifier_fit():
    n_in = 100
    xs, ys, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    size = [n_in, n_in / 4, 2]
    classifier = MLP(size=size, weights=None, learning_rate=0.2)
    classifier_score = classifier.fit(xs, ys, train_idx, valid_idx, n_epochs=100)
    assert classifier_score > 0.99


@use_logger_in_test
def test_classifier_infer():
    n_in = 100
    xs, ys, train_idx, valid_idx = gen_data(n_in=n_in, n_x1=300, n_x2=500)
    size = [n_in, n_in / 4, 2]
    classifier = MLP(size=size, weights=None)
    ey = classifier.infer(xs)
    np.testing.assert_array_equal(ey.shape, (len(ys), len(np.unique(ys))))
    assert 0 < np.sum(ey) < ey.shape[0] * ey.shape[1]