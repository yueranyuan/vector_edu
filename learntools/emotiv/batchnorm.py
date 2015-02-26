import theano
import theano.tensor as T
import numpy as np

from itertools import chain

from learntools.libs.logger import log_me
from learntools.libs.auc import auc
from learntools.model.mlp import MLP
from learntools.model.theano_utils import make_shared
from learntools.model import Model, gen_batches_by_size
from learntools.model.net import BatchNormLayer

class BatchNorm(Model):
    @log_me('...building BatchNorm')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
                 classifier_width=500, classifier_depth=1, rng_seed=42,
                 learning_rate=0.0002, **kwargs):
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

        self.train_idx = train_idx
        self.batch_size = batch_size
        #self.train_batches = gen_batches_by_size(train_idx, batch_size)
        self.valid_batches = gen_batches_by_size(valid_idx, 1)

        # 2: Connect the model
        rng = np.random.RandomState(rng_seed)
        self.rng = rng

        input_idxs = T.ivector('input_idxs')
        input_layer = self._xs[input_idxs]
        bn_updates, subnets = [], []
        train_layer, infer_layer, updates_layer = input_layer, input_layer, []

        n_in, n_out = input_size, classifier_width
        for i in xrange(classifier_depth):
          bn_layer = BatchNormLayer(rng, n_in, n_out)
          train_layer, infer_layer, updates_layer = bn_layer.instance(train_layer, infer_layer)
          bn_updates.extend(updates_layer)
          subnets.append(bn_layer)
          n_in, n_out = classifier_width, classifier_width

        # softmax
        bn_layer = BatchNormLayer(rng, n_in, n_out)
        train_pY, infer_pY, updates_layer = bn_layer.instance(train_layer, infer_layer, activation=T.nnet.softmax)
        bn_updates.extend(updates_layer)
        subnets.append(bn_layer)

        true_y = self._ys[input_idxs]
        true_y.name = 'true_y'

        # 3: Create theano functions
        loss = -T.mean(T.log(train_pY + 1e-8)[T.arange(input_idxs.shape[0]), true_y])
        loss.name = 'loss'
        cost = (
            loss
            + L1_reg * sum([net.L1 for net in subnets])
            + L2_reg * sum([net.L2_sqr for net in subnets])
        )
        cost.name = 'overall_cost'

        params = chain.from_iterable(net.params for net in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params] + bn_updates

        self._tf_infer = theano.function(inputs=[input_idxs], outputs=[loss, infer_pY[:, 1], input_idxs], allow_input_downcast=True)
        self._tf_train = theano.function(inputs=[input_idxs], outputs=[loss, train_pY[:, 1], input_idxs], allow_input_downcast=True, updates=update_parameters)
        self.subnets = subnets

    def evaluate(self, idxs, pred):
        y = self._ys.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(y[:len(pred)], pred, pos_label=1)

    def validate(self, idxs, **kwargs):
        return self._tf_infer(idxs)

    def train(self, idxs, **kwargs):
        return self._tf_train(idxs)

    @property
    def train_batches(self, **kwargs):
        shuffled_idx = self.rng.permutation(self.train_idx)

        return [shuffled_idx[begin : begin + self.batch_size] for begin in xrange(0, len(shuffled_idx), self.batch_size)]