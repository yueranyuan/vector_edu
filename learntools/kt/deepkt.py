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

from theano import config

config.exception_verbosity = 'high'


def _gen_batches(idxs, subjects, batch_size):
    '''divide row indicies for deepkt.

    divide indices into batches by subject ids and indices for each subject are
        further divided into sub batches by some minimal size. The first 2 rows of each
        subject are removed by necessity due to the recursive structure of the model

    Args:
        idxs (int[]): row indices
        subjects (int[]): list of subject ids corresponding to each row
            (could also be an EnumColumn). Subject ids must be pre-sorted.
        batch_size: the size of the subject's sub batches

    Returns:
        int[][]: list of batches

    Example:
        >>> _gen_batches(xrange(11), [1] * 6 + [2] * 5, 2)
        [[2, 3], [4, 5], [8, 9]]
    '''
    batches = gen_batches_by_keys(idxs, [subjects])
    batches = imap(lambda idxs: islice(idxs, 2, None), batches)
    sub_batches = imap(lambda idxs: gen_batches_by_size(list(idxs), batch_size), batches)
    batches = chain.from_iterable(sub_batches)
    batches = ifilter(lambda b: b, batches)
    batches = list(batches)
    return batches


class DeepKT(Model):
    '''a trainable, applyable model for deep kt
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    @log_me('...building deepkt')
    def __init__(self, prepared_data, L1_reg=0., L2_reg=0., dropout_p=0., learning_rate=0.02,
                 skill_vector_len=100, combiner_depth=1, combiner_width=200,
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

        ds, train_idx, valid_idx = prepared_data
        N = len(ds['correct'])
        eeg_vector_len = ds.get_data('eeg').shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['subject']))
        valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

        sorted_i = sorted(xrange(N), key=lambda i: (ds['subject'][i]))
        ds.reorder(sorted_i)

        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        train_idx = mask_to_idx(train_mask)
        valid_idx = mask_to_idx(valid_mask)
        base_indices = T.ivector('idx')

        skill_x = ds.get_data('skill')
        subject_x = ds.get_data('subject')
        correct_y = ds.get_data('correct')
        eeg_full = ds.get_data('eeg')

        # ###########
        # STEP2: connect up the model. See figures/vector_edu_model.png for diagram
        # TODO: make the above mentioned diagram

        # make a skill matrix containing skill vectors for each skill
        skill_matrix = make_shared(gen_word_matrix(ds.get_data('skill'),
                                                   ds['skill'].enum_pairs,
                                                   vector_length=skill_vector_len))

				# data preloaded into network
        skill_x = make_shared(skill_x, to_int=True, name='skill')
        correct_y = make_shared(correct_y, to_int=True, name='correct')
        eeg_full = make_shared(eeg_full, name='eeg')

        rng = np.random.RandomState(1234)
        t_dropout = T.scalar('dropout')
        skill_accumulator = make_shared(np.zeros((N, combiner_width)), name='skill_accumulator')

        # setup combiner component
        combiner_n_in = combiner_width + skill_vector_len + 1
        if previous_eeg_on:
            combiner_n_in += eeg_vector_len
        combiner = HiddenNetwork(
            rng=rng,
            n_in=combiner_n_in,
            size=[combiner_width] * combiner_depth,
            activation=rectifier,
            dropout=t_dropout
        )

        # setup main network component
        classifier_n_in = skill_vector_len
        if combiner_on:
            classifier_n_in += combiner_width
        if current_eeg_on:
            classifier_n_in += eeg_vector_len
        # final softmax classifier
        classifier = MLP(rng=rng,
                         n_in=classifier_n_in,
                         size=[main_net_width] * main_net_depth,
                         n_out=3,
                         dropout=t_dropout)

        # STEP 3.1 stuff that goes in scan
        current_skill = skill_matrix[skill_x[base_indices]]
        previous_skill = skill_matrix[skill_x[base_indices - 1]]
        previous_eeg_vector = eeg_full[base_indices - 1]
        current_eeg_vector = eeg_full[base_indices]

        # need to convert list of indices of 1,2 into [0],[1] columns
        correct_vectors = make_shared([[0], [1]])
        correct_feature = correct_vectors[correct_y[base_indices - 1] - 1]
        correct_feature.name = 'correct_feature'
        combiner_inputs = [skill_accumulator[base_indices - 2], previous_skill, correct_feature]
        if previous_eeg_on:
            combiner_inputs.append(previous_eeg_vector)
        t_combiner_inputs = T.concatenate(combiner_inputs, axis=1)
        t_combiner_inputs.name = 'combiner_inputs'
        combiner_out = combiner.instance(t_combiner_inputs)
        classifier_inputs = [current_skill]
        if combiner_on:
            classifier_inputs.append(combiner_out)
        if current_eeg_on:
            classifier_inputs.append(current_eeg_vector)
        # probability of y for each 0, 1, 2
        pY = classifier.instance(T.concatenate(classifier_inputs, axis=1))
        # ########
        # STEP3: create the theano functions to run the model

        y = correct_y[base_indices]
        loss = -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
        # used to help compute regularization terms
        subnets = [classifier]
        if combiner_on:
            subnets.append(combiner)
        cost = (
            loss
            + L1_reg * sum([n.L1 for n in subnets])
            + L2_reg * sum([n.L2_sqr for n in subnets])
        )

        # the same for both validation and training
        func_args = {
            'inputs': [base_indices],
            'outputs': [loss, pY[:, -2] - pY[:, -1], base_indices, pY, previous_eeg_vector],
            'on_unused_input': 'ignore',
            'allow_input_downcast': True,
        }

        # collect all theano updates
        params = chain.from_iterable(n.params for n in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params]
        # propagation of previous skill to next is computed as an update
        basic_updates = []
        if combiner_on:
            basic_updates += [(
                skill_accumulator,
                T.set_subtensor(skill_accumulator[base_indices - 1], combiner_out)
            )]
        # validation uses no dropout and previous skill propagation
        self._tf_valid = theano.function(
            updates=basic_updates,
            givens={t_dropout: 0.},
            **func_args)
        # training uses parameter updates plus previous skill propagation with dropout
        self._tf_train = theano.function(
            updates=update_parameters + basic_updates,
            givens={t_dropout: dropout_p},
            **func_args)
        self.train_batches = _gen_batches(train_idx, subject_x, batch_size)
        self.valid_batches = _gen_batches(valid_idx, subject_x, 1)
        self._correct_y = correct_y

    def evaluate(self, idxs, pred):
        '''scores the predictions of a given set of rows
        Args:
            idxs (int[]): the indices of the rows to be evaluated
            pred (float[]): the prediction for the label of that row
        Returns:
            float: an evaluation score (the higher the better)
        '''
        # _correct_y is int-casted, go to owner op (int-cast) to get shared variable
        # as first input and get its value without copying the value out
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
