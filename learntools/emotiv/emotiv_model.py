from itertools import imap, chain, groupby, islice, ifilter

import theano
import theano.tensor as T
import numpy as np

from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.data import gen_word_matrix
from learntools.libs.logger import log_me
from learntools.libs.auc import auc
from learntools.model.mlp import HiddenNetwork, MLP
from learntools.model.math import rectifier
from learntools.model.theano_utils import make_shared
from learntools.model import Model, gen_batches_by_keys, gen_batches_by_size
from learntools.libs.logger import gen_log_name, log_me, log, set_log_file

def _gen_batches(idxs, subjects, batch_size):
    return [idxs]

def _gen_train_valid_idxs(ds, cv_fold=0):
    validation_ratio = 0.1
    rng = np.random.RandomState(12345)
    rvec = rng.random_sample(len(ds))
    train_idx = mask_to_idx(rvec >= validation_ratio)
    valid_idx = mask_to_idx(rvec < validation_ratio)
    return train_idx, valid_idx

class EmotivModel(Model):
    '''a trainable, applyable model for deep kt
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    @log_me('...building model')
    def __init__(self, prepared_data, L1_reg=0., L2_reg=0., dropout_p=0., learning_rate=0.02,
                 cond_vector_len=100, combiner_depth=1, combiner_width=200,
                 main_net_depth=1, main_net_width=500, previous_eeg_on=1,
                 current_eeg_on=1, combiner_on=1, mutable_skill=1, valid_percentage=0.8,
                 batch_size=30, **kwargs):
        '''
        Args:
            prepared_data (tuple(Dataset, int[], int[])): a tuple that holds the data to be used,
                the row indices of the training set, and the row indices of the validation set
        '''
        # ##########
        # STEP1: order the data properly so that we can read from it sequentially
        # when training the model

        #ds, train_idx, valid_idx = prepared_data
        ds = prepared_data
        train_idx, valid_idx = _gen_train_valid_idxs(ds)
        log('len of train_idx=%d, valid_idx=%d' % (len(train_idx), len(valid_idx)), True)
        N = len(ds)
        eeg_vector_len = ds.get_data('eeg').shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['group']))
        valid_mask = idx_to_mask(valid_idx, len(ds['group']))

        sorted_i = sorted(xrange(N), key=lambda i: (ds['group'][i]))
        ds.reorder(sorted_i)

        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        train_idx = mask_to_idx(train_mask)
        valid_idx = mask_to_idx(valid_mask)
        base_indices = T.ivector('idx')

        cond_x = ds.get_data('condition')
        cond_count = len(np.unique(cond_x))
        group_x = ds.get_data('group')
        eeg_full = ds.get_data('eeg')

        rng = np.random.RandomState(1234)
        t_dropout = T.scalar('dropout')

        # ###########
        # STEP2: connect up the model. See figures/vector_edu_model.png for diagram
        # TODO: make the above mentioned diagram

        cond_matrix = make_shared(gen_word_matrix(ds.get_data('condition'),
                                                   ds['condition'].enum_pairs,
                                                   vector_length=cond_vector_len))

        cond_x = make_shared(cond_x, to_int=True)
        eeg_full = make_shared(eeg_full)
        combiner_n_in = eeg_full.shape.eval()[1]
        print combiner_n_in
        combiner = HiddenNetwork(
            rng=rng,
            n_in=combiner_n_in,
            size=[combiner_width] * combiner_depth,
            activation=rectifier,
            dropout=t_dropout
        )

        # setup main network component
        classifier = MLP(rng=rng,
                         n_in=combiner_width,
                         size=[main_net_width] * main_net_depth,
                         n_out=cond_count + 1,
                         dropout=t_dropout)

        # STEP 3.1 stuff that goes in scan
        current_cond = cond_matrix[cond_x[base_indices]]
        current_eeg_vector = eeg_full[base_indices]

        combiner_inputs = [current_eeg_vector]
        combiner_out = combiner.instance(T.concatenate(combiner_inputs, axis=1))
        classifier_inputs = [combiner_out]
        pY = classifier.instance(T.concatenate(classifier_inputs, axis=1))
        # ########
        # STEP3: create the theano functions to run the model

        y = cond_x[base_indices]
        loss = -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
        subnets = [combiner, classifier]
        cost = (
            loss +
            L1_reg * sum([n.L1 for n in subnets])
            + L2_reg * sum([n.L2_sqr for n in subnets])
        )

        func_args = {
            'inputs': [base_indices],
            'outputs': [loss, pY[:, -2] - pY[:, -1], base_indices, pY],
            'on_unused_input': 'ignore',
            'allow_input_downcast': True
        }

        params = chain.from_iterable(n.params for n in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params]
        self._tf_valid = theano.function(
            givens={t_dropout: 0.},
            **func_args)
        self._tf_train = theano.function(
            updates=update_parameters,
            givens={t_dropout: dropout_p},
            **func_args)
        self.train_batches = _gen_batches(train_idx, group_x, batch_size)
        self.valid_batches = _gen_batches(valid_idx, group_x, 1)
        self._correct_y = cond_x

    def evaluate(self, idxs, pred):
        '''scores the predictions of a given set of rows
        Args:
            idxs (int[]): the indices of the rows to be evaluated
            pred (float[]): the prediction for the label of that row
        Returns:
            float: an evaluation score (the higher the better)
        '''
        _y = self._correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y[:len(pred)], pred, pos_label=1)

    def train(self, idxs, **kwargs):
        '''perform one iteration of training on some indices
        Args:
            idxs (int[]): the indices of the rows to be used in training
        Returns:
            (float, float[], int[]): a tuple of the loss, the predictions over the rows,
                and the row indices
        '''
        res = self._tf_train(idxs)
        return res[:3]

    def validate(self, idxs, **kwargs):
        '''perform one iteration of validation
        Args:
            idxs (int[]): the indices of the rows to be used in validation
        Returns:
            (float, float[], int[]): a tuple of the loss, the predictions over the rows,
                and the row indices
        '''
        res = self._tf_valid(idxs)
        return res[:3]
