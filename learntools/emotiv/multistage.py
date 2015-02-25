"""
Multistage training.
"""

import theano
import theano.tensor as T
import numpy as np

from contextlib import contextmanager
from itertools import chain
from copy import deepcopy
import abc
import cPickle as pickle
import sys
import os

from learntools.libs.logger import log_me, log, set_log_file
from learntools.libs.auc import auc
from learntools.model.math import rectifier, sigmoid
from learntools.emotiv.data import segment_raw_data, filter_indices_by_condition
from learntools.data import cv_split


@log_me()
def run_multistage(task_num=0, dataset_name=None, conds=None, validation_p=0.1, snapshots=None, **kwargs):
    if snapshots is None:
        snapshots = []
    # load full dataset
    dataset = segment_raw_data(dataset_name, conds=None, **kwargs)
    Xs = dataset.get_data('eeg')
    ys = dataset.get_data('condition')
    n_in = Xs.shape[1]
    train_idx, valid_idx = cv_split(dataset, percent=validation_p, fold_index=task_num)

    # feature scaling
    scaler = FeatureScaler().fit(Xs[train_idx])

    stage = len(snapshots)

    # unsupervised training, first layer
    if stage > 0:
        with open(snapshots[0], 'rb') as f:
            autoencoder_params = pickle.load(f)
        autoencoder = Autoencoder(**autoencoder_params)
        autoencoder_score = autoencoder.evaluate(Xs[valid_idx])
    else:
        size = [n_in, n_in / 4]
        autoencoder = Autoencoder(size=size, weights=None, wrapper=scaler, **kwargs)
        autoencoder_score = autoencoder.fit(Xs, train_idx, valid_idx, **kwargs)
        log((autoencoder_score, autoencoder.parameters), True)

    # unsupervised training, second layer
    if stage > 1:
        with open(snapshots[1], 'rb') as f:
            autoencoder2_params = pickle.load(f)
        autoencoder2 = Autoencoder(**autoencoder2_params)
        autoencoder2_score = autoencoder2.evaluate(Xs[valid_idx])
    else:
        size = [n_in / 4, n_in / 16]
        autoencoder2 = Autoencoder(size=size, weights=None, wrapper=autoencoder, **kwargs)
        autoencoder2_score = autoencoder2.fit(Xs, train_idx, valid_idx, **kwargs)
        log((autoencoder2_score, autoencoder2.parameters), True)

    if stage > 2:
        with open(snapshots[2], 'rb') as f:
            autoencoder3_params = pickle.load(f)
        autoencoder3 = Autoencoder(**autoencoder3_params)
        autoencoder3_score = autoencoder3.evaluate(Xs[valid_idx])
    else:
        size = [n_in, n_in / 4, n_in / 16]
        W1, b_e1, b_d1 = autoencoder.parameters['weights']
        W2, b_e2, b_d2 = autoencoder2.parameters['weights']
        autoencoder3 = Autoencoder(size=size, weights=(W1 + W2, b_e1 + b_e2, b_d2 + b_d1), wrapper=scaler, learning_rate=5.0, **kwargs)
        autoencoder3_score = autoencoder3.fit(Xs, train_idx, valid_idx, **kwargs)
        log((autoencoder3_score, autoencoder3.parameters), True)

    # supervised training
    # narrow down to only the classes we are interested in using previous train/valid split
    cond_train_idx = filter_indices_by_condition(dataset, train_idx, conds)
    cond_valid_idx = filter_indices_by_condition(dataset, valid_idx, conds)
    size = [n_in, n_in / 4, n_in / 16, 2]
    W3, b_e3, b_d3 = autoencoder3.parameters['weights']
    classifier = Classifier(size=size, weights=(W3 + [None], b_e3 + [None]), wrapper=scaler, learning_rate=0.2, **kwargs)
    classifier_score = classifier.fit(Xs, ys, cond_train_idx, cond_valid_idx, **kwargs)
    log((classifier_score, classifier.parameters), True)


def generate_batches(rng, train_idx, batch_size):
    shuffled_idx = rng.permutation(train_idx)

    for begin in xrange(0, len(shuffled_idx), batch_size):
        yield shuffled_idx[begin : begin + batch_size]


@contextmanager
def np_printoptions(*args, **kwargs):
    """http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array"""
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


class Codec(object):
    """Interface for encode-decode pairs."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Set the parameters to best fit the data."""
        return self

    @abc.abstractmethod
    def encode(self, values, *args, **kwargs):
        """Encodes the value table."""
        return values

    @abc.abstractmethod
    def decode(self, values, *args, **kwargs):
        """Decodes an encoding."""
        return values

    def reconstruct(self, values, *args, **kwargs):
        """Applies the encoder, then the decoder."""
        return self.decode(self.encode(values, *args, **kwargs), *args, **kwargs)

    @abc.abstractproperty
    def parameters(self):
        """The parameter dictionary that can be used to instantiate the codec."""
        return {}

    def __repr__(self):
        """Serializable, unambiguous representation."""
        return "{}({})".format(self.__class__.__name__, ', '.join(k + '=' + repr(v) for k, v in self.parameters.items()))

    def __str__(self):
        return self.__repr__()

    def save_parameters(self, location):
        with open(location, 'wb') as f:
            pickle.dump(self.parameters, f, pickle.HIGHEST_PROTOCOL)


class Identity(Codec):
    def __init__(self):
        pass

    def fit(self):
        pass

    def encode(self, values):
        return values

    def decode(self, values):
        return values

    @property
    def parameters(self):
        return {}


class FeatureScaler(Codec):
    def __init__(self, min=0.0, max=1.0, epsilon=0.0, **kwargs):
        """Builds a scaler."""
        self.min = min
        self.max = max
        self.epsilon = epsilon

    def fit(self, data, **kwargs):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        self.epsilon = 1e-8
        return self

    def encode(self, values):
        return (values - self.min) / (self.max - self.min + self.epsilon)

    def decode(self, values):
        return values * (self.max - self.min + self.epsilon) + self.min

    @property
    def parameters(self):
        return {
            'min': self.min,
            'max': self.max,
            'epsilon': self.epsilon,
        }


class Autoencoder(Codec):
    def __init__(self, size=None, weights=None, wrapper=None, L1_reg=0.0, L2_reg=0.0, learning_rate=1.0, rng_state=None,
        batch_size=30, activation='rectifier', encoding_L1=0.0, **kwargs):
        """Builds an autoencoder.
        size: a list [input_dim (, layer_widths...), encoding_dim];
            input_dim: width of the input.
            layer_widths: widths of encoding layers; the decoder will be the reverse.
            encoding_dim: width of the middle encoding layer.
        weights: a tuple (Ws, b_e, b_d); if None, initialize randomly.
            Ws: a list of weights of the encoder going from input dimensions to encoding dimensions.
            b_e: a list of biases for the encoder, with dimensions 1 by layer_widths + [encoding_dim].
            b_d: a list of biases for the decoder, with dimensions 1 by reversed(layer_widths) + [input_dim].
        wrapper: an instantiated codec used to transform input values and output values (e.g. feature scaling).
        L1_reg: L1 regularization constant.
        L2_reg: L2 regularization constant.
        learning_rate: Learning rate.
        rng_state: numpy state tuple for rng.
        batch_size: size of each minibatch during training.
        activation: type of activation (as a string).
        encoding_L1: L1 regularization on the encoding layer.
        """
        rng = np.random.RandomState()
        if rng_state is not None:
            rng.set_state(rng_state)

        if activation == 'rectifier': # string for serializability
            activation_fn = rectifier
        else:
            raise ValueError

        if size is None or len(size) < 2:
            raise ValueError

        input_dim, layer_widths, encoding_dim = size[0], size[1:-1], size[-1]


        if weights is None:
            Ws, b_e, b_d = [None for i in size[1:]], [None for i in size[1:]], [None for i in size[1:]]
        else:
            Ws, b_e, b_d = weights

        for i, dim_in, dim_out in zip(xrange(len(Ws)), size, size[1:]):
            if Ws[i] is None:
                # Xavier initialization
                sigma2 = 2.0 / dim_in if activation == 'rectifier' else 2.0 / (dim_in + dim_out)
                Ws[i] = np.asarray(sigma2 * rng.randn(dim_in, dim_out), dtype=theano.config.floatX)
            if b_e[i] is None:
                b_e[i] = np.asarray(np.zeros((1, dim_out)), dtype=theano.config.floatX)
            if b_d[-(i+1)] is None:
                b_d[-(i+1)] = np.asarray(np.zeros((1, dim_in)), dtype=theano.config.floatX)

        if wrapper is None:
            wrapper = Identity()

        # construct the net
        transformed_input_vec = T.dmatrix('transformed_input_vec')
        layer = transformed_input_vec

        tf_Ws = []
        tf_bs = []

        # encoding layers
        for i, weight, bias in zip(xrange(1, len(Ws) + 1), Ws, b_e):
            tf_W = theano.shared(value=weight, name='W' + str(i), borrow=True)
            tf_b = theano.shared(value=bias, name='b' + str(i), borrow=True, broadcastable=(True, False))
            tf_Ws.append(tf_W)
            tf_bs.append(tf_b)
            layer = activation_fn(T.dot(layer, tf_W) + tf_b)

        encoding = layer

        # decoding layers
        for i, tf_W, bias in zip(xrange(len(Ws) + 1, 2 * len(Ws) + 1), reversed(tf_Ws), b_d):
            tf_b = theano.shared(value=bias, name='b' + str(i), borrow=True, broadcastable=(True, False))
            tf_bs.append(tf_b)
            layer = activation_fn(T.dot(layer, tf_W.T) + tf_b)

        transformed_reconstruction = layer

        # construct theano functions
        loss = T.mean(T.sqr(transformed_input_vec - transformed_reconstruction))
        L1 = sum(abs(tf_W).sum() for tf_W in tf_Ws)
        L2_sqr = sum((tf_W ** 2).sum() for tf_W in tf_Ws)
        encoding_reg = abs(encoding).sum() #(T.jacobian(T.flatten(encoding), transformed_input_vec) ** 2).sum()

        cost = loss + L1_reg * L1 + L2_reg * L2_sqr + encoding_L1 * encoding_reg

        params = tf_Ws + tf_bs
        updates = [(param, param - learning_rate * T.grad(cost, param)) for param in params]

        self._tf_reconstruct = theano.function(inputs=[transformed_input_vec], outputs=[transformed_reconstruction], allow_input_downcast=True)
        self._tf_encode = theano.function(inputs=[transformed_input_vec], outputs=[encoding], allow_input_downcast=True)
        self._tf_decode = theano.function(inputs=[encoding], outputs=[transformed_reconstruction], allow_input_downcast=True)
        self._tf_train = theano.function(inputs=[transformed_input_vec], outputs=[loss], allow_input_downcast=True, updates=updates)
        self._tf_evaluate = theano.function(inputs=[transformed_input_vec], outputs=[loss], allow_input_downcast=True)
        self._rng = rng

        # includes newly initialized values
        self.size = size
        self.weights = (Ws, b_e, b_d)
        self.wrapper = wrapper
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.encoding_L1 = encoding_L1

    def fit(self, dataset, train_idx, valid_idx, valid_frequency=0.2, **kwargs):
        """Performs training and sets the parameters of the model to those that minimize validation cost,
        and returns the best score achieved."""
        log("...fitting Autoencoder with size {}".format(self.size), True)

        epoch = 0
        valid_epoch = 0.0

        best_parameters = None
        best_cost = float('inf')

        while True:
            try:
                # stochastic gradient descent training
                cost = 0.0
                for batch in generate_batches(self._rng, train_idx, self.batch_size):
                    batch_cost = self.train(dataset[batch])
                    cost += len(batch) * batch_cost

                log("epoch {}, train cost {}".format(epoch, cost / len(train_idx)), True)

                # validation
                if int(valid_epoch + valid_frequency) != int(valid_epoch):
                    valid_epoch -= 1.0
                    valid_cost = self.evaluate(dataset[valid_idx])
                    log("epoch {}, validation cost {}".format(epoch, valid_cost), True)

                    if valid_cost < best_cost:
                        best_cost = valid_cost
                        best_parameters = deepcopy(self.parameters)

                valid_epoch += valid_frequency
                epoch += 1
            except KeyboardInterrupt:
                break

        import pdb; pdb.set_trace()

        # roll back
        self.__init__(**best_parameters)
        return best_cost

    def train(self, input_vec):
        """Performs one round of updates on input vectors within dataset."""
        return self._tf_train(self.wrapper.encode(input_vec))[0]

    def reconstruct(self, input_vec):
        """Returns predicted values on input vectors within dataset."""
        return self.wrapper.decode(self._tf_reconstruct(self.wrapper.encode(input_vec))[0])

    def evaluate(self, input_vec):
        """Returns reconstruction cost without L1/L2 normalization."""
        return self._tf_evaluate(self.wrapper.encode(input_vec))[0]

    def encode(self, input_vec):
        return self._tf_encode(self.wrapper.encode(input_vec))[0]

    def decode(self, encoded_vec):
        return self.wrapper.decode(self._tf_decode(encoded_vec)[0])

    @property
    def parameters(self):
        return {
            'size': self.size,
            'weights': self.weights,
            'wrapper': self.wrapper,
            'L1_reg': self.L1_reg,
            'L2_reg': self.L2_reg,
            'learning_rate': self.learning_rate,
            'rng_state': self._rng.get_state(),
            'batch_size': self.batch_size,
            'activation': self.activation,
            'encoding_L1': self.encoding_L1,
        }


class Classifier(object):
    def __init__(self, size=None, weights=None, wrapper=None, L1_reg=0.0, L2_reg=0.0, learning_rate=0.5, rng_state=None, batch_size=30, activation='rectifier', **kwargs):
        """Multilayer perceptron with softmax.
        size: A list [input_dim (, layer_widths...), num_classes]
            input_dim: width of the input vector.
            layer_widths: dimensions of the hidden layers.
            num_classes: number of classes. Input for classes will be 0 to num_classes-1.
        weights: A tuple (W, b)
            Ws: a list of weights of the feedforward network.
            bs: a list of biases of the feedforward network.
        wrapper: an instantiated codec used to transform input values.
        L1_reg: L1 regularization constant.
        L2_reg: L2 regularization constant.
        learning_rate: learning rate.
        rng_state: numpy state tuple for rng.
        batch_size: batch size.
        activation: type of activation (as a string).
        """
        rng = np.random.RandomState()
        if rng_state is not None:
            rng.set_state(rng_state)

        if activation == 'rectifier':
            activation_fn = rectifier
        else:
            raise ValueError

        if size is None or len(size) < 2:
            raise ValueError

        if weights is None:
            Ws, bs = [None for i in size[1:]], [None for i in size[1:]]
        else:
            Ws, bs = weights

        for i, dim_in, dim_out in zip(xrange(len(Ws)), size, size[1:]):
            if Ws[i] is None:
                # Xavier initialization
                sigma2 = 2.0 / dim_in if activation == 'rectifier' else 2.0 / (dim_in + dim_out)
                Ws[i] = np.asarray(sigma2 * rng.randn(dim_in, dim_out), dtype=theano.config.floatX)
            if bs[i] is None:
                bs[i] = np.asarray(np.zeros((1, dim_out)), dtype=theano.config.floatX)

        if wrapper is None:
            wrapper = Identity()

        # construct the net
        transformed_input_vec = T.dmatrix('transformed_input_vec')
        layer = transformed_input_vec

        tf_Ws, tf_bs = [], []

        for i, weight, bias in zip(xrange(len(Ws)), Ws, bs):
            tf_W = theano.shared(value=weight, name='W' + str(i), borrow=True)
            tf_b = theano.shared(value=bias, name='b' + str(i), borrow=True, broadcastable=(True, False))
            tf_Ws.append(tf_W)
            tf_bs.append(tf_b)
            if i < len(Ws) - 1:
                layer = activation_fn(T.dot(layer, tf_W) + tf_b)
            else:
                est_probs = T.nnet.softmax(T.dot(layer, tf_W) + tf_b)

        # construct theano functions
        true_y = T.ivector('true_y')
        # negative log likelihood
        loss = -T.mean(T.log(est_probs)[T.arange(true_y.shape[0]), true_y])
        L1 = sum(abs(tf_W).sum() for tf_W in tf_Ws)
        L2_sqr = sum((tf_W ** 2).sum() for tf_W in tf_Ws)

        cost = loss + L1_reg * L1 + L2_reg * L2_sqr

        params = tf_Ws + tf_bs
        updates = [(param, param - learning_rate * T.grad(cost, param)) for param in params]

        self._tf_train = theano.function(inputs=[transformed_input_vec, true_y], outputs=[loss], allow_input_downcast=True, updates=updates)
        self._tf_infer = theano.function(inputs=[transformed_input_vec], outputs=[est_probs], allow_input_downcast=True)
        self._tf_evaluate = theano.function(inputs=[transformed_input_vec, true_y], outputs=[loss], allow_input_downcast=True)
        self._rng = rng

        self.size = size
        self.weights = (Ws, bs)
        self.wrapper = wrapper
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation

    def fit(self, Xs, ys, train_idx, valid_idx, valid_frequency=0.2, binary=True, **kwargs):
        """Performs training and sets the parameters of the model to those that minimize validation cost,
        and returns the best score achieved.
        binary: whether or not the classes are binary classes. If so, show auc score instead of confusion matrix score."""
        log("...fitting Classifier with size {}".format(self.size), True)

        epoch = 0
        valid_epoch = 0.0

        best_parameters = None
        best_score = 0.0

        while True:
            try:
                # stochastic gradient descent training
                for batch in generate_batches(self._rng, train_idx, self.batch_size):
                    self.train(Xs[batch], ys[batch])

                score = self.evaluate(Xs[train_idx], ys[train_idx], binary=binary)

                log("epoch {}, train score {}".format(epoch, score), True)

                # validation
                if int(valid_epoch + valid_frequency) != int(valid_epoch):
                    valid_epoch -= 1.0
                    valid_score = self.evaluate(Xs[valid_idx], ys[valid_idx], binary=binary)
                    log("epoch {}, validation score {}".format(epoch, valid_score), True)

                    if valid_score > best_score:
                        best_score = valid_score
                        best_parameters = deepcopy(self.parameters)

                valid_epoch += valid_frequency
                epoch += 1
            except KeyboardInterrupt:
                break

        import pdb; pdb.set_trace()

        # roll back
        self.__init__(**best_parameters)
        return best_score

    def train(self, input_vec, true_y):
        return self._tf_train(self.wrapper.encode(input_vec), true_y)[0]

    def evaluate(self, input_vec, true_y, binary=True):
        if binary:
            return self.auc(input_vec, true_y)
        else:
            cm, d = self.confusion(input_vec, true_y)
            return d
        #return self._tf_evaluate(self.wrapper.encode(input_vec), true_y)[0]

    def confusion(self, input_vec, true_y):
        """Returns a normalized confusion matrix and the sum along the diagonal."""
        est_probs = self._tf_infer(self.wrapper.encode(input_vec))[0]
        cm = np.zeros((self.size[-1], self.size[-1]))
        for row in xrange(input_vec.shape[0]):
            cm[true_y[row], :] += est_probs[row] / len(input_vec)
        return (cm, sum(cm.diagonal()))

    def auc(self, input_vec, true_y, pos_label=1):
        """Returns auc score for binary classification."""
        return auc(true_y, self.infer(input_vec)[:, pos_label], pos_label=pos_label)

    def infer(self, input_vec):
        """Returns probability vector for each class."""
        return self._tf_infer(self.wrapper.encode(input_vec))[0]

    def predict(self, input_vec):
        """Returns the class values with the highest probability."""
        return np.argmax(self.infer(input_vec), axis=1)

    @property
    def parameters(self):
        return {
            'size': self.size,
            'weights': self.weights,
            'wrapper': self.wrapper,
            'L1_reg': self.L1_reg,
            'L2_reg': self.L2_reg,
            'learning_rate': self.learning_rate,
            'rng_state': self._rng.get_state(),
            'batch_size': self.batch_size,
            'activation': self.activation,
        }

    def save_parameters(self, location):
        with open(location, 'wb') as f:
            pickle.dump(self.parameters, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    set_log_file(os.devnull)
    run_multistage(task_num=0, dataset_name=sys.argv[1], snapshots=sys.argv[2:], conds=['EyesOpen', 'EyesClosed'])