import theano
import theano.tensor as T
import numpy as np

from learntools.model.net import TrainableNetwork, Codec, HiddenLayer


class DecoderLayer(HiddenLayer):
    def __init__(self, t_W, inp=None, rng_state=None, b=None,
                 activation='rectifier', name='decoderlayer'):
        super(HiddenLayer, self).__init__(inp=inp, name=name, rng_state=rng_state)

        ########
        # STEP 0: load initializing values

        self.activation = activation

        self.t_W = t_W.T
        self.t_W.name = self.subname('W')

        if b is None:
            b = np.zeros((self.W.shape[1],), dtype=theano.config.floatX)
        self.t_b = theano.shared(value=b, name=self.subname('b'), borrow=True)

        self.activation = activation
        self.params = [self.t_b]

        if self.input is None:
            self.input = T.dmatrix(self.subname('input'))

    @property
    def W(self):
        return self.t_W.owner.inputs[0].get_value(borrow=True).T

    @property
    def b(self):
        return self.t_b.get_value(borrow=True)

    @property
    def weights(self):
        return [(self.W, self.b)]

    def instance(self, x, **kwargs):
        lin_output = T.dot(x, self.t_W) + self.t_b
        return self.activation_fn(lin_output)


class Autoencoder(Codec, TrainableNetwork):
    def __init__(self, name="autoencoder", inp=None, size=None, weights=None, rng_state=None,
                 activation='rectifier', **kwargs):
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
        super(Autoencoder, self).__init__(name=name, rng_state=rng_state, inp=inp)

        self.input = T.dmatrix(self.subname("input"))

        if weights is None:
            weights = [(None, None) for i in xrange(len(size))]
        if size is None:
            size = [w.shape[0] for w, b in weights]

        # build encoder network
        self.encoder = HiddenNetwork(name=self.subname("encoder"), inp=inp, size=size, weights=weights[:-1],
                                     rng_state=rng_state, activation=activation)

        # build decoder network
        t_decoder_W = self.encoder.layers[-1].t_W
        decoder_b = weights[-1][1]
        self.decoder = DecoderLayer(name=self.subname("decoder"), t_W=t_decoder_W, rng_state=rng_state,
                                    b=decoder_b, activation=activation)
        self.components = [self.encoder, self.decoder]
        self.true_output = T.dmatrix('reconstruction_target')
        self.compile()

    @property
    def loss(self):
        if not hasattr(self, '_loss'):
            self._loss = T.mean(T.sqr(self.true_output - self.output))
        return self._loss

    def instance(self, x, dropout=None):
        self.encoding = self.encoder.instance(x, dropout=dropout)
        output = self.decoder.instance(self.encoding)
        return output

    def compile(self):
        super(Autoencoder, self).compile()
        self._tf_encode = theano.function(inputs=[self.input], outputs=[self.encoding], allow_input_downcast=True)
        self._tf_decode = theano.function(inputs=[self.encoding], outputs=[self.output], allow_input_downcast=True)

    def reconstruct(self, input_vec):
        """Returns predicted values on input vectors within dataset."""
        return self.infer(input_vec)

    def evaluate(self, input_vec, y, **kwargs):
        """Returns reconstruction cost without L1/L2 normalization."""
        return self._tf_evaluate(input_vec, y)[0]

    def encode(self, input_vec):
        return self._tf_encode(input_vec)[0]

    def decode(self, encoded_vec):
        return self._tf_decode(encoded_vec)[0]

    def fit(self, Xs, train_idx, valid_idx, binary=False, **kwargs):
        return super(Autoencoder, self).fit(Xs=Xs, ys=Xs, train_idx=train_idx, valid_idx=valid_idx, binary=False, **kwargs)

    def __pickle__(self):
        from learntools.libs.utils import combine_dict
        super_pickle = super(Autoencoder, self).__pickle__()
        my_pickle = {
            'weights': self.encoder.weights + self.decoder.weights,
            'activation': self.encoder.activation
        }
        return combine_dict(super_pickle, my_pickle)

    @property
    def weights(self):
        return self.encoder.weights + self.decoder.weights

from learntools.model.net import HiddenNetwork
from learntools.model.logistic import LogisticRegression


class MLP(TrainableNetwork):
    def __init__(self, name='classifier', inp=None, size=None, weights=None, rng_state=None,
                 activation='rectifier', **kwargs):
        """Multilayer perceptron with softmax.
        size: A list [input_dim (, layer_widths...), num_classes]
            input_dim: width of the input vector.
            layer_widths: dimensions of the hidden layers.
            num_classes: number of classes. Input for classes will be 0 to num_classes-1.
        weights: A tuple (W, b)
            Ws: a list of weights of the feedforward network.
            bs: a list of biases of the feedforward network.
        rng_state: numpy state tuple for rng.
        activation: type of activation (as a string).
        """
        super(MLP, self).__init__(inp=inp, name=name, rng_state=rng_state)

        if weights is None:
            weights = [(None, None) for i in xrange(len(size) - 1)]

        if self.input is None:
            self.input = T.dmatrix(self.subname('input'))
        self.true_output = T.ivector('true_y')

        # setup hidden network
        hidden_size = None if size is None else size[:-1]
        self.hidden = HiddenNetwork(size=hidden_size, inp=inp, weights=weights[:-1], activation=activation,
                                    rng_state=rng_state)
        # setup top layer logistic unit
        logistic_W, logistic_b = weights[-1]
        if size is None:
            logistic_n_in = logistic_n_out = None
        else:
            logistic_n_in, logistic_n_out = size[-2:]
        self.logistic = LogisticRegression(n_in=logistic_n_in, n_out=logistic_n_out, W=logistic_W, b=logistic_b,
                                           rng_state=rng_state)
        self.components = [self.hidden, self.logistic]

        # compile theano functions
        self.compile()

    def instance(self, x):
        hidden_out = self.hidden.instance(x)
        est_probs = self.logistic.instance(hidden_out)
        return est_probs

    @property
    def n_outputs(self):
        Ws, _ = self.weights
        return Ws[-1].shape[1]

    def evaluate(self, input_vec, true_y, binary=True):
        if binary:
            return self.auc(input_vec, true_y)
        else:
            cm, d = self.confusion(input_vec, true_y)
            return d

    def confusion(self, input_vec, true_y):
        """Returns a normalized confusion matrix and the sum along the diagonal."""
        est_probs = self.infer(input_vec)
        cm = np.zeros((self.n_outputs, self.n_outputs))
        for row in xrange(input_vec.shape[0]):
            cm[true_y[row], :] += est_probs[row] / len(input_vec)
        return (cm, sum(cm.diagonal()))

    def __pickle__(self):
        from learntools.libs.utils import combine_dict
        super_pickle = super(MLP, self).__pickle__()
        my_pickle = {
            'weights': self.hidden.weights + [(self.logistic.W, self.logistic.b)],
            'activation': self.hidden.activation,
        }
        return combine_dict(super_pickle, my_pickle)