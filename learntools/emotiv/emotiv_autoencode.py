import theano
import theano.tensor as T
import numpy as np

from itertools import chain

from learntools.libs.logger import log_me
from learntools.model.mlp import MLP
from learntools.model.net import HiddenNetwork
from learntools.model.theano_utils import make_shared
from learntools.model import gen_batches_by_size
from learntools.emotiv.base import BaseEmotiv


class AutoencodeEmotiv(BaseEmotiv):
    @log_me('...building AutoencodeEmotiv')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
                 classifier_width=500, classifier_depth=1, rng_seed=42, dropout_p=0.5,
                 learning_rate=0.02, autoencoder_weight=0.3, **kwargs):
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
        input_size = ds.get_data('eeg').shape[1]

        self._xs = make_shared(ds.get_data('eeg'), name='eeg')
        self._ys = make_shared(ds.get_data('condition'), to_int=True, name='condition')

        self.train_batches = gen_batches_by_size(train_idx, batch_size)
        self.valid_batches = gen_batches_by_size(valid_idx, 1)

        # 2: Connect the model
        rng = np.random.RandomState(rng_seed)
        t_dropout = T.scalar('dropout')

        bottom_layer = HiddenNetwork(rng=rng,
                                     n_in=input_size,
                                     size=[classifier_width],
                                     dropout=t_dropout,
                                     name='bottomlayer')

        autoencoder = HiddenNetwork(rng=rng, n_in=classifier_width, size=[input_size], name='autoencoder')
        top_layers = MLP(rng=rng,
                         n_in=classifier_width,
                         size=[classifier_width] * (classifier_depth - 1),
                         n_out=2,
                         dropout=t_dropout,
                         name='toplayers')

        input_idxs = T.ivector('input_idxs')
        classifier_input = self._xs[input_idxs]
        classifier_input.name = 'classifier_input'
        bottom_layer_output = bottom_layer.instance(classifier_input)
        reconstructed_input = autoencoder.instance(bottom_layer_output)
        pY = top_layers.instance(bottom_layer_output)
        true_y = self._ys[input_idxs]
        true_y.name = 'true_y'

        # 3: Create theano functions
        loss = -T.mean(T.log(pY)[T.arange(input_idxs.shape[0]), true_y])
        loss.name = 'loss'
        autoencoder_loss = -T.mean(T.sqr(classifier_input - reconstructed_input))
        autoencoder_loss.name = 'autoencoder_loss'
        autoencoder_loss_scaled = autoencoder_weight * autoencoder_loss / input_size
        autoencoder_loss_scaled.name = 'autoencoder_loss_scaled'
        subnets = [bottom_layer, autoencoder, top_layers]
        cost = (
            loss
            + autoencoder_loss_scaled
            + L1_reg * sum([net.L1 for net in subnets])
            + L2_reg * sum([net.L2_sqr for net in subnets])
        )
        cost.name = 'overall_cost'

        reconstruction_networks = [bottom_layer, autoencoder]
        reconstruction_cost = (
            autoencoder_loss_scaled
            + L1_reg * sum([net.L1 for net in reconstruction_networks])
            + L2_reg * sum([net.L2_sqr for net in reconstruction_networks])
        )

        func_args = {
            'inputs': [input_idxs],
            'outputs': [loss, pY[:, 1] - pY[:, 0], input_idxs],
            'allow_input_downcast': True,
        }
        params = chain.from_iterable(net.params for net in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params]

        reconstruction_params = chain.from_iterable(net.params for net in reconstruction_networks)
        update_reconstruction_params = [(param, param - learning_rate * T.grad(reconstruction_cost, param))
                                        for param in reconstruction_params]

        self._tf_valid = theano.function(
            updates=update_reconstruction_params,
            givens={t_dropout: 0.},
            **func_args)
        self._tf_train = theano.function(
            updates=update_parameters,
            givens={t_dropout: dropout_p},
            **func_args)