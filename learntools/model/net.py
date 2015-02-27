import abc
import cPickle as pickle
from copy import deepcopy

import numpy as np
import theano
import theano.tensor as T

from learntools.model.math import rectifier, sigmoid
from learntools.libs.logger import log
from learntools.libs.auc import auc
from learntools.libs.utils import transpose
import sys


class NetworkComponent(object):
    """Abstract network object that is not meant to be used. Holds some convenience functions
    and the signature for NetworkComponents"""
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, name, rng_state=None, inp=None, *args, **kwargs):
        """create all theano variables at initialization

        Args:
            name (str): the name of the network. Use subname() to generate names for
                all theano variables which are a part of this network so that debug will
                display an appropriate name for the variable that indicates its place in the
                network hierarchy
        """
        self.name = name

        self._rng = np.random.RandomState()
        if rng_state is not None:
            self._rng.set_state(rng_state)

        if inp is not None:
            self.input = inp
        else:
            self.input = None

    def compile(self):
        self._tf_infer = theano.function(inputs=[self.input], outputs=[self.output], allow_input_downcast=True)

    @abc.abstractmethod
    def instance(self, x, **kwargs):
        """generate the theano variable for the output of this network given the input x

        Args:
            x: a theano variable that represents the input to this network

        Returns:
            a theano variable that represents the output of this network
        """
        pass

    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = np.random.RandomState()
        return self._rng

    @property
    def output(self):
        if not hasattr(self, '_output'):
            self._output = self.instance(self.input)
        return self._output

    def subname(self, suffix):
        '''generate a name for a theano variable. Make sure all theano variables that are a
        part of this network are named using names generated by this function

        Args:
            suffix (str): the desired name of the theano variable

        Returns:
            (str): the full name of a theano variable with the name of this network appended'''
        return '{root}_{suffix}'.format(root=self.name, suffix=suffix)

    @property
    def L1(self):
        '''L1 regularization value'''
        if hasattr(self, '_L1'):
            return self._L1
        if hasattr(self, 'components') and len(self.components) > 0:
            self._L1 = sum([c.L1 for c in self.components])
        else:
            self._L1 = sum([abs(param).sum() for param in self.params])
        self._L1.name = self.subname('L1')
        return self._L1

    @L1.setter
    def L1(self, L1):
        self._L1 = L1
        self._L1.name = self.subname('L1')

    @property
    def L2_sqr(self):
        '''L2 regularization value'''
        if hasattr(self, '_L2_sqr'):
            return self._L2_sqr
        if hasattr(self, 'components'):
            self._L2_sqr = sum([c.L2_sqr for c in self.components])
        else:
            self._L2_sqr = sum([(param ** 2).sum() for param in self.params])
        self._L2_sqr.name = self.subname('L2_sqr')
        return self._L2_sqr

    @L2_sqr.setter
    def L2_sqr(self, L2_sqr):
        self._L2_sqr = L2_sqr
        self._L2_sqr.name = self.subname('L2_sqr')

    @property
    def params(self):
        '''all differentiable theano variables of this network'''
        if hasattr(self, '_params'):
            return self._params
        self.params = sum([c.params for c in self.components], [])
        return self.params

    @params.setter
    def params(self, params):
        self._params = params

    def infer(self, input_vec):
        """Returns probability vector for each class."""
        return self._tf_infer(input_vec)[0]

    def __pickle__(self):
        return {
            'name': self.name,
            'rng_state': self.rng.get_state(),
        }

    def _serialize_to_file(self, file):
        pickle.dump(self.serialize(), file, pickle.HIGHEST_PROTOCOL)

    def serialize(self, file=None):
        """Serialize network component

        Args:
            file (str or file optional): the file to write this network component to

        Returns:
            (dict): a dictionary of the keyword arguments to initialize this network component
        """
        # serialize parameters out to a file
        if file is not None:
            if isinstance(file, str):  # file is the string location of the file
                with open(file, 'wb') as f:
                    self._serialize_to_file(f)
            else:  # file is the file object itself
                self._serialize_to_file(file)

        return self.__pickle__()

    @classmethod
    def deserialize(cls, parameters=None, data_file=None, inp=None):
        """Deserialize the network component

        Args:
            parameters (dict optional): the keyword dictionary of arguments to instantiate this class
            file (str or file optional): a string location or a fileio object where the parameters are pickled

        Returns:
            NetworkComponent: the deserialized network component
        """
        # deserialize parameters from a file
        if data_file is not None:
            if isinstance(data_file, str):  # data_file is the string location of the file
                with open(data_file, 'wb') as f:
                    parameters = pickle.load(f)
            else:  # data_file is the file object itself
                parameters = pickle.load(data_file)

        return cls(inp=inp, **parameters)


class TrainableNetwork(NetworkComponent):
    """Abstract Network that has an objective function and can be trained and applied
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, name='trainable_network', inp=None, rng_state=None, **kwargs):
        super(TrainableNetwork, self).__init__(inp=inp, name=name, rng_state=rng_state)

    @abc.abstractmethod
    def instance(self, x, **kwargs):
        pass

    @property
    def loss(self):
        if hasattr(self, '_loss'):
            return self._loss
        self._loss = -T.mean(T.log(self.output)[T.arange(self.true_output.shape[0]), self.true_output])
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss
        self._loss.name = self.subname('loss')

    def compile(self, additional_updates=None):
        """ compile theano functions
        """
        if additional_updates is None:
            additional_updates = []

        self.t_L1_reg = T.fscalar('L1_reg')
        self.t_L2_reg = T.fscalar('L2_reg')
        self.t_learning_rate = T.fscalar('learning_rate')
        cost = self.loss + self.t_L1_reg * self.L1 + self.t_L2_reg * self.L2_sqr

        parameter_updates = [(param, param - self.t_learning_rate * T.grad(cost, param)) for param in self.params]
        updates = parameter_updates + additional_updates

        self._tf_train = theano.function(inputs=[self.input, self.true_output, self.t_L1_reg, self.t_L2_reg, self.t_learning_rate],
                                         outputs=[self.loss], allow_input_downcast=True, updates=updates)
        self._tf_infer = theano.function(inputs=[self.input], outputs=[self.output], allow_input_downcast=True)
        self._tf_evaluate = theano.function(inputs=[self.input, self.true_output], outputs=[self.loss],
                                            allow_input_downcast=True)

    def fit(self, Xs, ys, train_idx, valid_idx, valid_frequency=0.2, n_epochs=1000, binary=True,
            L1_reg=0.0, L2_reg=0.0, learning_rate=0.5, batch_size=30, **kwargs):
        """Performs training and sets the parameters of the model to those that minimize validation cost,
        and returns the best score achieved.
        binary: whether or not the classes are binary classes. If so, show auc score instead of confusion matrix score.
        L1_reg: L1 regularization constant.
        L2_reg: L2 regularization constant.
        learning_rate: learning rate.
        batch_size: batch size.
        """
        log("...fitting Classifier", True)

        valid_epoch = 0.0

        best_parameters = None
        best_score = 0.0

        for epoch_i in xrange(n_epochs):
            try:
                # stochastic gradient descent training
                for batch in generate_batches(self._rng, train_idx, batch_size):
                    self.train(Xs[batch], ys[batch], L1_reg, L2_reg, learning_rate)

                score = self.evaluate(Xs[train_idx], ys[train_idx], binary=binary)

                log("epoch {}, train score {}".format(epoch_i, score), True)

                # validation
                if int(valid_epoch + valid_frequency) != int(valid_epoch):
                    valid_epoch -= 1.0
                    valid_score = self.evaluate(Xs[valid_idx], ys[valid_idx], binary=binary)
                    log("epoch {}, validation score {}".format(epoch_i, valid_score), True)

                    if valid_score > best_score:
                        best_score = valid_score
                        best_parameters = deepcopy(self.__pickle__())

                valid_epoch += valid_frequency
            except KeyboardInterrupt:
                break

        # import pdb; pdb.set_trace()
        # self.__init__(**best_parameters)  # this is causing a lot of problems with theano redeclarations

        # roll back
        return best_score

    def train(self, input_vec, true_y, L1_reg=0.0, L2_reg=0.0, learning_rate=0.5):
        return self._tf_train(input_vec, true_y, L1_reg, L2_reg, learning_rate)[0]

    def evaluate(self, input_vec, true_y):
        return self.auc(input_vec, true_y)

    def auc(self, input_vec, true_y, pos_label=1):
        """Returns auc score for binary classification."""
        return auc(true_y, self.infer(input_vec)[:, pos_label], pos_label=pos_label)

    def predict(self, input_vec):
        """Returns the class values with the highest probability."""
        return np.argmax(self.infer(input_vec), axis=1)


class Codec(NetworkComponent):
    """Abstract network that can encode and decode"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def instance(self, x, **kwargs):
        pass

    def encode(self, values, *args, **kwargs):
        """Encodes the value table."""
        return self.infer(values)

    @abc.abstractmethod
    def decode(self, values, *args, **kwargs):
        """Decodes an encoding."""
        pass

    def reconstruct(self, values, *args, **kwargs):
        """Applies the encoder, then the decoder."""
        return self.decode(self.encode(values, *args, **kwargs), *args, **kwargs)


def generate_batches(rng, train_idx, batch_size):
    shuffled_idx = rng.permutation(train_idx)

    for begin in xrange(0, len(shuffled_idx), batch_size):
        yield shuffled_idx[begin : begin + batch_size]


# inspired by https://github.com/mdenil/dropout/blob/master/mlp.py
class HiddenLayer(NetworkComponent):
    def __init__(self, inp=None, n_in=None, n_out=None, rng_state=None, W=None, b=None,
                 activation='rectifier', name='hiddenlayer'):
        super(HiddenLayer, self).__init__(inp=inp, name=name, rng_state=rng_state)

        ########
        # STEP 0: load initializing values

        self.activation = activation

        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
            self.rng.randint(999999))

        if W is None:
            sigma2 = 2.0 / n_in if activation == 'rectifier' else 2.0 / (n_in + n_out)
            W = np.asarray(sigma2 * self.rng.randn(n_in, n_out), dtype=theano.config.floatX)
        self.t_W = theano.shared(value=W, name=self.subname('W'), borrow=True)

        if n_out is None:
            n_out = W.shape[1]

        if b is None:
            b = np.zeros((n_out,), dtype=theano.config.floatX)
        self.t_b = theano.shared(value=b, name=self.subname('b'), borrow=True)

        self.activation = activation
        self.params = [self.t_W, self.t_b]

        if self.input is None:
            self.input = T.dmatrix(self.subname('input'))

    @property
    def W(self):
        return self.t_W.get_value(borrow=True)

    @property
    def b(self):
        return self.t_b.get_value(borrow=True)

    def instance(self, x, dropout=None, **kwargs):
        # dropouts
        dropout = dropout or 0.
        mask = self.srng.binomial(n=1, p=1 - dropout, size=x.shape)
        # cast because int * float32 = float64 which does not run on GPU
        x = x * T.cast(mask, theano.config.floatX)
        lin_output = (T.dot(x, self.t_W) + self.t_b) * (1 / (1 - dropout))
        return self.activation_fn(lin_output)

    @property
    def activation_fn(self):
        if self.activation == 'rectifier':
            return rectifier
        elif self.activation == 'sigmoid':
            return sigmoid
        else:
            raise ValueError

    def __pickle__(self):
        from learntools.libs.utils import combine_dict
        super_pickle = super(HiddenLayer, self).__pickle__()
        my_pickle = {
            'W': self.W,
            'b': self.b,
            'activation': self.activation,
        }
        return combine_dict(super_pickle, my_pickle)


class HiddenNetwork(NetworkComponent):
    def __init__(self, size=None, weights=None, rng_state=None, inp=None, name='hiddennetwork',
                 activation='rectifier', **kwargs):
        super(HiddenNetwork, self).__init__(inp=inp, name=name, rng_state=rng_state)

        ########
        # STEP 0: load initializing values

        self.activation = activation

        min_layers = 1
        # derive size from weights
        if size is None and weights is None:
            raise ValueError
        elif size is None and weights is not None:
            # derive size from weights
            Ws, bs = transpose(weights)
            if len(Ws) < min_layers:
                raise ValueError
            size = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
        elif size is not None and weights is None:
            if len(size) < min_layers:
                raise ValueError
            Ws, bs = [None for i in size[1:]], [None for i in size[1:]]
        else:
            Ws, bs = transpose(weights)

        ########
        # STEP 1: initialize network

        self.layers = []
        for i, (n_in_, n_out_, W, b) in enumerate(zip(size, size[1:], Ws, bs)):
            self.layers.append(HiddenLayer(rng_state=rng_state,
                                           n_in=n_in_,
                                           n_out=n_out_,
                                           W=W,
                                           b=b,
                                           name=self.subname('layer{i}'.format(i=i)),
                                           activation=activation,
                                           **kwargs))
        self.components = self.layers

        if self.input is None:
            self.input = T.dmatrix(self.subname('input'))

    def instance(self, x, dropout=None, **kwargs):
        inp = x
        for layer in self.layers:
            inp = layer.instance(inp, dropout)
        return inp

    @property
    def weights(self):
        return [(layer.W, layer.b) for layer in self.layers]

    def __pickle__(self):
        from learntools.libs.utils import combine_dict
        super_pickle = super(HiddenNetwork, self).__pickle__()
        my_pickle = {
            'weights': self.weights,
            'activation': self.activation,
        }
        return combine_dict(super_pickle, my_pickle)


class BatchNormLayer(HiddenLayer):
    def __init__(self, rng, n_in, n_out, W=None, b=None, beta=None, gamma=None, alpha=0.999, mean=None, variance=None, activation=rectifier, name='batchnormlayer'):
        """alpha is the exponential moving average falloff multiplier"""
        super(BatchNormLayer, self).__init__(rng, n_in, n_out, W=W, b=b, activation=activation, name=name)
        if beta is None:
            beta_values = np.zeros((n_out,), dtype=theano.config.floatX)
            beta = theano.shared(value=beta_values, name=self.subname('beta'), borrow=True)

        if gamma is None:
            gamma_values = np.ones((n_out,), dtype=theano.config.floatX)
            gamma = theano.shared(value=gamma_values, name=self.subname('gamma'), borrow=True)

        if mean is None:
            mean_values = np.zeros((n_out,), dtype=theano.config.floatX)
            mean = theano.shared(value=mean_values, name=self.subname('mean'), borrow=True)

        if variance is None:
            variance_values = np.ones((n_out,), dtype=theano.config.floatX)
            variance = theano.shared(value=variance_values, name=self.subname('variance'), borrow=True)

        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.mean = mean
        self.variance = variance

        self.activation = activation
        self.params.extend([beta, gamma])

    def instance(self, train_x, infer_x, epsilon=1e-8, **kwargs):
        """Returns (train_output, inference_output, statistics_updates)"""
        train_lin_output = T.dot(train_x, self.W) + self.b
        batch_mean = T.mean(train_lin_output)
        offset_output = train_lin_output - batch_mean
        batch_var = T.var(offset_output)
        normalized_lin_output =  offset_output / T.sqrt(batch_var + epsilon)
        train_output = self.activation(self.gamma * normalized_lin_output + self.beta)

        infer_lin_output = T.dot(infer_x, self.W) + self.b
        sd = T.sqrt(self.variance + epsilon)
        inference_output = self.activation(self.gamma / sd * infer_lin_output + (self.beta - (self.gamma * self.mean) / sd))
        # exponential moving average for batch mean/variance
        statistics_updates = [
            (self.mean, self.alpha * self.mean + (1.0 - self.alpha) * batch_mean),
            (self.variance, self.alpha * self.variance + (1.0 - self.alpha) * batch_var)
        ]
        return (train_output, inference_output, statistics_updates)
