import theano
import theano.tensor as T
import numpy as np

from itertools import chain

from learntools.libs.logger import log_me
from learntools.libs.auc import auc
from learntools.model.mlp import HiddenNetwork, MLP
from learntools.model.math import rectifier
from learntools.model.theano_utils import make_shared
from learntools.model import Model, gen_batches_by_size

class BaseEmotiv(Model):
    @log_me('...building BaseEmotiv')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
        middle_width=100, middle_depth=1, classifier_width=500, classifier_depth=1,
        rng_seed=42, dropout_p=0.5, learning_rate=0.02, **kwargs):
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

        self._xs = make_shared(ds.get_data('eeg'))
        self._ys = make_shared(ds.get_data('condition'), to_int=True)

        self.train_batches = gen_batches_by_size(train_idx, batch_size)
        self.valid_batches = gen_batches_by_size(valid_idx, 1)

        # 2: Connect the model
        rng = np.random.RandomState(rng_seed)
        t_dropout = T.scalar('dropout')

        middle_layer = HiddenNetwork(
            rng=rng,
            n_in=input_size,
            size=[middle_width] * middle_depth,
            activation=rectifier,
            dropout=t_dropout
        )

        classifier = MLP(rng=rng,
                                         n_in=middle_width,
                                         size=[classifier_width] * classifier_depth,
                                         n_out=2,
                                         dropout=t_dropout)

        input_idxs = T.ivector('input_idxs')
        middle_out = middle_layer.instance(self._xs[input_idxs])
        pY = classifier.instance(middle_out)
        true_y = self._ys[input_idxs]

        # 3: Create theano functions
        loss = -T.mean(T.log(pY)[T.arange(input_idxs.shape[0]), true_y])
        subnets = [middle_layer, classifier]
        cost = (
            loss
            + L1_reg * sum([net.L1 for net in subnets])
            + L2_reg * sum([net.L2_sqr for net in subnets])
        )

        func_args = {
            'inputs': [input_idxs],
            'outputs': [loss, pY[:, 1] - pY[:, 0], input_idxs],
            'allow_input_downcast': True,
        }
        params = chain.from_iterable(net.params for net in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                                                 for param in params]

        self._tf_valid = theano.function(givens={t_dropout: 0.}, **func_args)
        self._tf_train = theano.function(
            updates=update_parameters,
            givens={t_dropout: dropout_p},
            **func_args)

    def evaluate(self, idxs, pred):
        y = self._ys.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(y[:len(pred)], pred, pos_label=1)

    def validate(self, idxs, **kwargs):
        return self._tf_valid(idxs)

    def train(self, idxs, **kwargs):
        return self._tf_train(idxs)