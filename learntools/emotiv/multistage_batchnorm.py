from __future__ import division

from itertools import chain, izip

import theano
import theano.tensor as T
import numpy as np

from learntools.libs.logger import log_me
from learntools.model.theano_utils import make_shared
from learntools.model import gen_batches_by_size
from learntools.model.net import AutoencodingBatchNormLayer
from learntools.emotiv.batchnorm import BatchNorm


class AutoencodingBatchNorm(BatchNorm):
    @log_me('...building AutoencodingBatchNorm')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
                 classifier_width=500, classifier_depth=2, rng_seed=42,
                 learning_rate=0.0002, dropout_p=0.0, **kwargs):
        """
        Args:
            prepared_data : (Dataset, [int], [int])
                a tuple that holds the data to be used, the row indices of the
                training set, and the row indices of the validation set
            batch_size : int
                The size of the batches used to train
        """

        classifier_width = 500
        classifier_depth = 1
        dropout_p = 0.1

        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data
        input_size = ds.get_data('eeg').shape[1]

        self._xs = make_shared(ds.get_data('eeg'), name='eeg')
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

        n_in, n_out = input_size, classifier_width
        train_recon_losses, infer_recon_losses = [], []
        for i in xrange(classifier_depth):
            bn_layer = AutoencodingBatchNormLayer(n_in=n_in, n_out=n_out)
            train_in, infer_in = train_out, infer_out
            train_in.name = 'train_in{}'.format(i)
            infer_in.name = 'infer_in{}'.format(i)
            train_out, infer_out, updates_layer, train_recon, infer_recon = bn_layer.instance(train_in, infer_in, dropout=t_dropout)
            bn_updates.extend(updates_layer)
            subnets.append(bn_layer)

            train_recon_loss = T.mean(T.sqr(train_in - train_recon))
            train_recon_losses.append(train_recon_loss)
            infer_recon_loss = T.mean(T.sqr(infer_in - infer_recon))
            infer_recon_loss.name = 'infer_loss{}'.format(i)
            infer_recon_losses.append(infer_recon_loss)
            n_in, n_out = classifier_width, classifier_width

        train_pY, infer_pY = train_recon, infer_recon

        # 3: Create theano functions
        train_loss = T.sum(train_recon_losses)
        train_loss.name = 'train_loss'
        infer_loss = T.sum(infer_recon_losses)
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
        self._tf_train = theano.function(inputs=[input_idxs],
                                         outputs=[train_loss, train_pY, input_idxs],
                                         givens=[(t_dropout, dropout_p)],
                                         allow_input_downcast=True, updates=update_parameters)
        self.subnets = subnets

    def train_full(self, **kwargs):
        return super(AutoencodingBatchNorm, self).train_full(train_with_loss=True, **kwargs)