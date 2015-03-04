import theano
import theano.tensor as T
import numpy as np

from itertools import chain, izip

from learntools.libs.logger import log_me
from learntools.libs.auc import auc
from learntools.model.mlp import ConvolutionalMLP
from learntools.model.theano_utils import make_shared
from learntools.model import Model, gen_batches_by_size
from learntools.model.math import sigmoid


class ConvEmotiv(Model):
    @log_me('...building ConvEmotiv')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
                 field_width=20, ds_factor=2, rng_seed=42, dropout_p=0.5,
                 learning_rate=0.02, **kwargs):
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
        self.valid_batches = gen_batches_by_size(valid_idx, batch_size)

        # 2: Connect the model
        rng = np.random.RandomState(rng_seed)
        t_dropout = T.scalar('dropout')

        classifier = ConvolutionalMLP(rng=rng,
                         n_in=input_size,
                         size=[input_size],
                         n_out=2,
                         field_width=field_width,
                         ds_factor=ds_factor,
                         dropout=t_dropout)

        input_idxs = T.ivector('input_idxs')
        classifier_input = self._xs[input_idxs]
        classifier_input.name = 'classifier_input'
        pY = classifier.instance(classifier_input)
        true_y = self._ys[input_idxs]
        true_y.name = 'true_y'

        # 3: Create theano functions
        loss = -T.mean(T.log(pY)[T.arange(input_idxs.shape[0]), true_y])
        loss.name = 'loss'
        subnets = [classifier]
        cost = (
            loss
            + L1_reg * sum([net.L1 for net in subnets])
            + L2_reg * sum([net.L2_sqr for net in subnets])
        )
        cost.name = 'overall_cost'

        # compute parameter updates
        training_updates = []
        params = list(chain.from_iterable(net.params for net in subnets))
        raw_deltas = [T.grad(cost, param) for param in params]
        if momentum > 0:
            old_deltas = [shared_zeros_like(p) for p in params]
            deltas = [momentum * old_delta + raw_delta for old_delta, raw_delta in izip(old_deltas, raw_deltas)]
            update_momentum = [(old_delta, delta) for old_delta, delta in izip(old_deltas, deltas)]
            training_updates += update_momentum
        else:
            deltas = raw_deltas
        update_parameters = [(param, param - learning_rate * delta)
                             for param, delta in izip(params, deltas)]
        training_updates += update_parameters

        common_args = {
            'inputs': [input_idxs],
            'outputs': [loss, pY[:, 1] - pY[:, 0], input_idxs],
            'allow_input_downcast': True,
        }
        self._tf_valid = theano.function(givens={t_dropout: 0.}, **common_args)
        self._tf_train = theano.function(
            updates=training_updates,
            givens={t_dropout: dropout_p},
            **common_args)

    def evaluate(self, idxs, pred):
        y = self._ys.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(y[:len(pred)], pred, pos_label=1)

    def validate(self, idxs, **kwargs):
        res = self._tf_valid(idxs)
        return res[:3]

    def train(self, idxs, **kwargs):
        res = self._tf_train(idxs)
        return res[:3]
