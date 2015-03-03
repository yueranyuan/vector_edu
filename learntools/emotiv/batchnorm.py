from __future__ import division

from itertools import chain, izip

import theano
import theano.tensor as T
import numpy as np

from learntools.libs.logger import log_me
from learntools.libs.auc import auc
from learntools.model.theano_utils import make_shared
from learntools.model import Model, gen_batches_by_size
from learntools.model.net import BatchNormLayer, AutoencodingBatchNormLayer, TrainableNetwork
from learntools.libs.utils import max_idx


class BatchNorm(Model):
    @log_me('...building BatchNorm')
    def __init__(self, prepared_data, batch_size=30, L1_reg=0., L2_reg=0.,
                 classifier_width=500, classifier_depth=2, rng_seed=42,
                 learning_rate=0.0002, dropout_p=0.0, recon_loss_alpha=0.0, **kwargs):
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
        output_size = len(np.unique(ds.get_data('condition')))

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
        bn_updates, subnets = [], []
        train_layer, infer_layer, updates_layer = input_layer, input_layer, []

        n_in, n_out = input_size, classifier_width
        recon_losses = []
        for i in xrange(classifier_depth):
            bn_layer = AutoencodingBatchNormLayer(n_in=n_in, n_out=n_out)
            train_layer, infer_layer, updates_layer, train_recon, infer_recon = bn_layer.instance(train_layer, infer_layer, dropout=t_dropout)
            bn_updates.extend(updates_layer)
            subnets.append(bn_layer)
            recon_loss = T.mean(T.sqr(train_layer - train_recon))
            recon_losses.append(recon_loss)
            n_in, n_out = classifier_width, classifier_width

        # softmax
        bn_layer = BatchNormLayer(n_in=classifier_width, n_out=output_size, activation='softmax')
        train_pY, infer_pY, updates_layer = bn_layer.instance(train_layer, infer_layer)
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
            # + recon_loss_alpha * T.sum(recon_losses)
        )
        cost.name = 'overall_cost'

        params = chain.from_iterable(net.params for net in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params] + bn_updates

        self._tf_infer = theano.function(inputs=[input_idxs],
                                         outputs=[loss, infer_pY[:, 1] - infer_pY[:, 0], input_idxs],
                                         givens=[(t_dropout, 0.0)],
                                         allow_input_downcast=True)
        self._tf_train = theano.function(inputs=[input_idxs],
                                         outputs=[loss, train_pY[:, 1] - train_pY[:, 0], input_idxs],
                                         givens=[(t_dropout, dropout_p)],
                                         allow_input_downcast=True, updates=update_parameters)
        self.subnets = subnets

    def validate(self, idxs, **kwargs):
        return self._tf_infer(idxs)

    def train(self, idxs, **kwargs):
        res = self._tf_train(idxs)
        return res[:3]

    @property
    def train_batches(self, **kwargs):
        shuffled_idx = self.rng.permutation(self.train_idx)
        return [shuffled_idx[begin: begin + self.batch_size] for begin in xrange(0, len(shuffled_idx), self.batch_size)]

    def train_evaluate(self, idxs, preds, *args, **kwargs):
        y = self._ys.owner.inputs[0].get_value(borrow=True)[idxs]
        best_pred_cutoff = find_best_cutoff(y, preds)

        # gradually change the prediction cutoff to avoid instability
        PRED_CUTOFF_ALPHA = 0.2
        if hasattr(self, 'pred_cutoff'):
            self.pred_cutoff = (1.0 - PRED_CUTOFF_ALPHA) * self.pred_cutoff + PRED_CUTOFF_ALPHA * best_pred_cutoff
        else:
            self.pred_cutoff = best_pred_cutoff

        return acc_by_cutoff(y, preds, self.pred_cutoff)

    def valid_evaluate(self, idxs, preds, *args, **kwargs):
        y = self._ys.owner.inputs[0].get_value(borrow=True)[idxs]
        best_acc = acc_by_cutoff(y, preds, find_best_cutoff(y, preds))
        _auc = auc(y[:len(preds)], preds, pos_label=1)
        print("validation auc: {auc}, best cutoff accuracy: {best_acc}".format(auc=_auc, best_acc=best_acc))
        return acc_by_cutoff(y, preds, self.pred_cutoff)


def acc_by_cutoff(y, preds, cutoff):
    """Compute accuracy given a certain prediction cutoff

    Examples:
    >>> acc_by_cutoff([0, 0, 1, 1], [0.2, 0.4, 0.7, 0.9], cutoff=0.3)
    0.75
    >>> acc_by_cutoff([0, 0, 1, 1], [0.2, 0.4, 0.7, 0.9], cutoff=0.5)
    1.0
    >>> acc_by_cutoff([0, 0, 1, 1], [0.2, 0.4, 0.7, 0.9], cutoff=0.1)
    0.5
    """
    ey = np.greater_equal(preds, cutoff)
    correct = sum(np.equal(y[:len(ey)], ey))
    return correct / len(ey)


def find_best_cutoff(y, preds):
    """ Find the prediction score cutoff that results in the highest prediction accuracy

    Examples:
    >>> find_best_cutoff([0, 0, 0, 1, 1, 1], [0.3, 0.5, 0.8, 0.7, 0.9, 0.9])
    0.7
    >>> find_best_cutoff([0, 1, 0, 1, 0, 1], [0.5, 0.9, 0.8, 0.7, 0.3, 0.9])
    0.7
    >>> ys = np.asarray([0] * 5000 + [1] * 5000, dtype='int32')
    >>> preds = np.random.rand(10000)
    >>> slow_best_cutoff = preds[max_idx[[acc_by_cutoff(ys, preds, cutoff) for cutoff in preds]][0]]
    >>> acc_by_cutoff(slow_best_cutoff) == acc_by_cutoff(find_best_cutoff(ys, preds))
    True
    """
    N = len(preds)
    if N == 0:
        raise Exception("trying to find best cutoff for an empty predictions array")
    sorted_pred_idxs = sorted(range(N), key=lambda (i): preds[i])

    # use Dynamic Programming to fill up the correctness table.
    # table for sum of correctness to the left of the current idx
    correct_left_table = [0] * N
    for i in xrange(1, N):
        correct = 1 if y[sorted_pred_idxs[i - 1]] == 0 else 0
        if i != 1:
            correct += correct_left_table[i - 1]
        correct_left_table[i] = correct
    # table for correctness to the right of the current idx
    correct_right_table = [0] * N
    for i in xrange(N - 1, -1, -1):
        correct = 1 if y[sorted_pred_idxs[i]] == 1 else 0
        if i != N - 1:
            correct += correct_right_table[i + 1]
        correct_right_table[i] = correct

    # find the best sum of left and right
    correct = [left + right for left, right in izip(correct_left_table, correct_right_table)]
    best_cutoff_i, best_correct = max_idx(correct)
    best_cutoff = preds[sorted_pred_idxs[best_cutoff_i]]

    return best_cutoff

'''
class BatchNormClassifier(TrainableNetwork):
    """ This is working but somehow isn't generating the same performance. I'll have to come back to this and figure it out.
    """
    @log_me('...building BatchNorm')
    def __init__(self, n_in, n_out, classifier_width=500, classifier_depth=2, inp=None,
                 name="batchNormClassifier", rng_state=None, **kwargs):
        """
        Args:
            prepared_data : (Dataset, [int], [int])
                a tuple that holds the data to be used, the row indices of the
                training set, and the row indices of the validation set
            batch_size : int
                The size of the batches used to train
        """
        super(BatchNormClassifier, self).__init__(name=name, rng_state=rng_state, inp=inp)
        # 1: Organize data into batches

        self.bn_updates, self.layers = [], []
        if self.input is None:
            self.input = T.imatrix(self.subname('input'))

        # build inner layers
        _n_in, _n_out = n_in, classifier_width
        for i in xrange(classifier_depth):
            bn_layer = BatchNormLayer(n_in=_n_in, n_out=_n_out)
            self.layers.append(bn_layer)
            _n_in, _n_out = classifier_width, classifier_width

        # build top softmax layer
        bn_layer = BatchNormLayer(n_in=classifier_width, n_out=n_out, activation='softmax')
        self.layers.append(bn_layer)

        self.true_output = T.ivector('true_output')

        self.components = self.layers
        self.output, self.output_inferred, self.bn_updates = self.instance(self.input)
        self.compile()

    def instance(self, x, **kwargs):
        train_layer, infer_layer, updates_layer = x, x, []
        bn_updates = []
        for layer in self.layers:
            train_layer, infer_layer, updates_layer = layer.instance(train_layer, infer_layer)
            bn_updates.extend(updates_layer)

        return train_layer, infer_layer, bn_updates

    def compile(self):
        super(BatchNormClassifier, self).compile()

        # some functions are wrong and need to be recompiled
        self._tf_infer = theano.function(inputs=[self.input], outputs=[self.output_inferred], allow_input_downcast=True)
        self._tf_train = theano.function(inputs=[self.input, self.true_output, self.t_L1_reg, self.t_L2_reg, self.t_learning_rate],
                                         outputs=[self.loss], allow_input_downcast=True, updates=self.parameter_updates + self.bn_updates)
'''