from __future__ import division

from itertools import chain, izip
import cPickle as pickle

import theano
import theano.tensor as T
import numpy as np

from learntools.libs.logger import log_me, log
from learntools.model.theano_utils import make_shared
from learntools.model import gen_batches_by_size
from learntools.model.net import BatchNormLayer, DecodingBatchNormLayer
from learntools.emotiv.batchnorm import BatchNorm
from learntools.libs.utils import normalize_table


def run(full_data=None, prepared_data=None, classifier_depth=2, **kwargs):
    previous_autoencoder_params = None
    for i in xrange(classifier_depth):
        autoencoder = AutoencodingBatchNorm(prepared_data=full_data, classifier_depth=i + 1,
                                            serialized=previous_autoencoder_params, **kwargs)
        _, autoencoder_params = autoencoder.train_full(**kwargs)
        previous_autoencoder_params = autoencoder_params[:-1]  # discard the decoder layer

    classifier2 = BatchNorm(prepared_data=prepared_data, serialized=previous_autoencoder_params,
                            classifier_depth=classifier_depth, **kwargs)
    classifier2.train_full(**kwargs)


def tune(prepared_data=None, weight_file=None, **kwargs):
    log('loading weight file {}'.format(weight_file))
    with open(weight_file, 'rb') as f:
        weights = pickle.load(f)
    classifier = BatchNorm(prepared_data=prepared_data, serialized=weights, **kwargs)
    classifier.train_full(**kwargs)


def pretrain(log_name=None, full_data=None, classifier_depth=2, **kwargs):
    classifier_depth = 1
    previous_autoencoder_params = None
    for i in xrange(classifier_depth):
        # discard the decoder layer
        previous_lower_layers = previous_autoencoder_params[:-1] if previous_autoencoder_params else None
        autoencoder = AutoencodingBatchNorm(prepared_data=full_data, classifier_depth=i + 1,
                                            serialized=previous_lower_layers, **kwargs)
        _, autoencoder_params = autoencoder.train_full(**kwargs)
        previous_autoencoder_params = autoencoder_params
    pickle.dump(previous_autoencoder_params, open("{log_name}.weights".format(log_name=log_name), "wb"))
    return previous_autoencoder_params


class AutoencodingBatchNorm(BatchNorm):
    @log_me('...building AutoencodingBatchNorm')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
                 classifier_width=500, classifier_depth=None, rng_seed=42,
                 learning_rate=0.0002, dropout_p=0.0, serialized=None, **kwargs):
        """
        Args:
            prepared_data : (Dataset, [int], [int])
                a tuple that holds the data to be used, the row indices of the
                training set, and the row indices of the validation set
            batch_size : int
                The size of the batches used to train
        """

        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data
        xs = ds.get_data('eeg')
        xs = normalize_table(xs)
        input_size = xs.shape[1]

        self._xs = make_shared(xs, name='eeg')
        self._ys = make_shared(ds.get_data('condition'), to_int=True, name='condition')

        self.train_idx = train_idx
        self.batch_size = batch_size
        self.valid_batches = gen_batches_by_size(valid_idx, 1)

        # 2: Connect the model
        rng = np.random.RandomState(rng_seed)
        self.rng = rng

        t_dropout = T.dscalar('dropout')
        input_idxs = T.ivector('input_idxs')
        input_layer = self._xs[input_idxs]
        input_layer.name = "input"
        bn_updates, subnets = [], []
        train_out, infer_out, updates_layer = input_layer, input_layer, []

        if serialized is None:
            serialized = []
        if classifier_depth is not None:
            serialized += [{}] * (classifier_depth + 1 - len(serialized))  # pad serialized subnets

        n_in, n_out = input_size, classifier_width
        self.layers = []
        for i, net_params in enumerate(serialized[:-1]):
            bn_layer = BatchNormLayer(n_in=n_in, n_out=n_out, **net_params)
            train_in, infer_in = train_out, infer_out
            train_in.name = 'train_in{}'.format(i)
            infer_in.name = 'infer_in{}'.format(i)
            train_out, infer_out, updates_layer = bn_layer.instance(train_in, infer_in, dropout=t_dropout)
            self.layers.append(bn_layer)
            # only train the top layer
            if i == len(serialized) - 1:
                bn_updates.extend(updates_layer)
                subnets.append(bn_layer)
            if i == 0:
                encoding = T.concatenate((infer_out, infer_in), axis=1)
            n_in, n_out = classifier_width, classifier_width

        if serialized[-1] is None:
            decoder_params = {}
        else:
            decoder_params = serialized[-1]
        bn_decoder = DecodingBatchNormLayer(encoder_W=bn_layer.t_W, **decoder_params)
        train_recon, infer_recon, updates_layer = bn_decoder.instance(train_out, infer_out)
        bn_updates.extend(updates_layer)
        self.layers.append(bn_decoder)
        subnets.append(bn_decoder)
        train_recon_loss = T.mean(T.sqr(train_in - train_recon))
        infer_recon_loss = T.mean(T.sqr(infer_in - infer_recon))
        encoding = T.concatenate((infer_recon, infer_in, input_layer), axis=1)

        train_pY, infer_pY = train_recon, infer_recon

        # 3: Create theano functions
        train_loss = train_recon_loss
        train_loss.name = 'train_loss'
        infer_loss = infer_recon_loss
        infer_loss.name = 'infer_loss'
        cost = (
            train_loss
            + L1_reg * sum([net.L1 for net in subnets])
            + L2_reg * sum([net.L2_sqr for net in subnets])
        )
        cost.name = 'overall_cost'

        params = chain.from_iterable(net.params for net in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params] + bn_updates

        self._tf_infer = theano.function(inputs=[input_idxs],
                                         outputs=[infer_loss, infer_pY, input_idxs],
                                         allow_input_downcast=True)
        self._tf_encode = theano.function(inputs=[input_idxs],
                                          outputs=encoding,
                                          allow_input_downcast=True)
        self._tf_train = theano.function(inputs=[input_idxs],
                                         outputs=[train_loss, train_pY, input_idxs],
                                         givens=[(t_dropout, dropout_p)],
                                         allow_input_downcast=True, updates=update_parameters)
        self.subnets = subnets

    def train_full(self, **kwargs):
        return super(AutoencodingBatchNorm, self).train_full(train_with_loss=True, **kwargs)

    def serialize(self):
        return [net.serialize() for net in self.layers]

    def encode(self, x):
        return self._tf_encode(x)
