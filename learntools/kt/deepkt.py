from itertools import imap, chain, groupby, islice

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
from learntools.model import Model, gen_batches


class DeepKT(Model):
    def __init__(self, prepared_data, L1_reg=0., L2_reg=0., dropout_p=0., learning_rate=0.02,
                 skill_vector_len=100, combiner_depth=1, combiner_width=200,
                 main_net_depth=1, main_net_width=500, previous_eeg_on=1,
                 current_eeg_on=1, combiner_on=1, mutable_skill=1, valid_percentage=0.8,
                 batch_size=30, **kwargs):
        # ##########
        # STEP1: order the data properly so that we can read from it sequentially
        # when training the model

        ds, train_idx, valid_idx = prepared_data
        N = len(ds['correct'])
        eeg_vector_len = ds['eeg'].shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['subject']))
        valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

        sorted_i = sorted(xrange(N), key=lambda i: (ds['subject'][i]))
        ds.reorder(sorted_i)

        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        train_idx = mask_to_idx(train_mask)
        valid_idx = mask_to_idx(valid_mask)
        base_indices = T.ivector('idx')

        skill_x = ds['skill']
        subject_x = ds['subject']
        correct_y = ds['correct']
        eeg_full = ds['eeg']

        # ###########
        # STEP2: connect up the model. See figures/vector_edu_model.png for diagram
        # TODO: make the above mentioned diagram

        skill_matrix = make_shared(gen_word_matrix(ds['skill'],
                                                   ds.get_column('skill').enum_pairs,
                                                   vector_length=skill_vector_len))

        skill_x = make_shared(skill_x, to_int=True)
        correct_y = make_shared(correct_y, to_int=True)
        eeg_full = make_shared(eeg_full)

        rng = np.random.RandomState(1234)
        t_dropout = T.scalar('dropout')
        skill_accumulator = make_shared(np.zeros((N, combiner_width)))

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
        classifier = MLP(rng=rng,
                         n_in=classifier_n_in,
                         size=[main_net_width] * main_net_depth,
                         n_out=3,
                         dropout=t_dropout)

        # STEP 3.1 stuff that goes in scan
        current_skill = skill_matrix[skill_x[base_indices]]
        previous_skill = skill_matrix[skill_x[base_indices - 1]]
        previous_eeg_vector = eeg_full[base_indices]
        current_eeg_vector = eeg_full[base_indices]

        correct_vectors = make_shared([[0], [1]])
        correct_feature = correct_vectors[correct_y[base_indices - 1] - 1]
        combiner_inputs = [skill_accumulator[base_indices - 2], previous_skill, correct_feature]
        if previous_eeg_on:
            combiner_inputs.append(previous_eeg_vector)
        combiner_out = combiner.instance(T.concatenate(combiner_inputs, axis=1))
        classifier_inputs = [current_skill]
        if combiner_on:
            classifier_inputs.append(combiner_out)
        if current_eeg_on:
            classifier_inputs.append(current_eeg_vector)
        pY = classifier.instance(T.concatenate(classifier_inputs, axis=1))
        # ########
        # STEP3: create the theano functions to run the model

        y = correct_y[base_indices]
        loss = -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
        subnets = [classifier]
        if combiner_on:
            subnets.append(combiner)
        cost = (
            loss
            + L1_reg * sum([n.L1 for n in subnets])
            + L2_reg * sum([n.L2_sqr for n in subnets])
        )

        func_args = {
            'inputs': [base_indices],
            'outputs': [loss, pY[:, -2] - pY[:, -1], base_indices, pY, previous_eeg_vector],
            'on_unused_input': 'ignore',
            'allow_input_downcast': True
        }

        params = chain.from_iterable(n.params for n in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params]
        basic_updates = []
        if combiner_on:
            basic_updates += [(
                skill_accumulator,
                T.set_subtensor(skill_accumulator[base_indices - 1], combiner_out)
            )]
        self.tf_valid = theano.function(
            updates=basic_updates,
            givens={t_dropout: 0.},
            **func_args)
        self.tf_train = theano.function(
            updates=update_parameters + basic_updates,
            givens={t_dropout: dropout_p},
            **func_args)
        self.train_batches = gen_batches(train_idx, [subject_x], batch_size)
        self.valid_batches = gen_batches(valid_idx, [subject_x], 1)
        self.correct_y = correct_y

    def evaluate(self, idxs, pred):
        _y = self.correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y[:len(pred)], pred, pos_label=1)

    def train(self, idxs, **kwargs):
        res = self.tf_train(idxs)
        return res[:3]

    def validate(self, idxs, **kwargs):
        res = self.tf_valid(idxs)
        return res[:3]


@log_me('... building the model')
def build_model(prepared_data, L1_reg=0., L2_reg=0., dropout_p=0., learning_rate=0.02,
                skill_vector_len=100, combiner_depth=1, combiner_width=200,
                main_net_depth=1, main_net_width=500, previous_eeg_on=1,
                current_eeg_on=1, combiner_on=1, mutable_skill=1, valid_percentage=0.8,
                batch_size=30, **kwargs):
    # ##########
    # STEP1: order the data properly so that we can read from it sequentially
    # when training the model

    ds, train_idx, valid_idx = prepared_data
    N = len(ds['correct'])
    eeg_vector_len = ds['eeg'].shape[1]
    train_mask = idx_to_mask(train_idx, len(ds['subject']))
    valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

    sorted_i = sorted(xrange(N), key=lambda i: (ds['subject'][i], ds['start_time'][i]))
    ds.reorder(sorted_i)
    train_mask = train_mask[sorted_i]
    valid_mask = valid_mask[sorted_i]
    train_idx = np.nonzero(train_mask)[0]
    valid_idx = np.nonzero(valid_mask)[0]
    base_indices = T.ivector('idx')

    skill_x = ds['skill']
    subject_x = ds['subject']
    correct_y = ds['correct']
    eeg_full = ds['eeg']

    # ###########
    # STEP2: connect up the model. See figures/vector_edu_model.png for diagram
    # TODO: make the above mentioned diagram
    skill_matrix = make_shared(gen_word_matrix(ds['skill'],
                                               ds.get_column('skill').enum_pairs,
                                               vector_length=skill_vector_len))

    skill_x = make_shared(skill_x, to_int=True)
    correct_y = make_shared(correct_y, to_int=True)
    eeg_full = make_shared(eeg_full)

    rng = np.random.RandomState(1234)
    t_dropout = T.scalar('dropout')
    skill_accumulator = make_shared(np.zeros((N, combiner_width)))

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
    classifier = MLP(rng=rng,
                     n_in=classifier_n_in,
                     size=[main_net_width] * main_net_depth,
                     n_out=2,
                     dropout=t_dropout)

    # STEP 3.1 stuff that goes in scan
    current_skill = skill_matrix[skill_x[base_indices]]
    previous_skill = skill_matrix[skill_x[base_indices - 1]]
    previous_eeg_vector = eeg_full[base_indices]
    current_eeg_vector = eeg_full[base_indices]

    correct_vectors = make_shared([[0], [1]])
    correct_feature = correct_vectors[correct_y[base_indices - 1] - 1]
    combiner_inputs = [skill_accumulator[base_indices - 2], previous_skill, correct_feature]
    if previous_eeg_on:
        combiner_inputs.append(previous_eeg_vector)
    combiner_out = combiner.instance(T.concatenate(combiner_inputs, axis=1))
    classifier_inputs = [current_skill]
    if combiner_on:
        classifier_inputs.append(combiner_out)
    if current_eeg_on:
        classifier_inputs.append(current_eeg_vector)
    pY = classifier.instance(T.concatenate(classifier_inputs, axis=1))
    # ########
    # STEP3: create the theano functions to run the model

    y = correct_y[base_indices]
    loss = -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
    subnets = [classifier]
    if combiner_on:
        subnets.append(combiner)
    cost = (
        loss
        + L1_reg * sum([n.L1 for n in subnets])
        + L2_reg * sum([n.L2_sqr for n in subnets])
    )

    func_args = {
        'inputs': [base_indices],
        'outputs': [loss, pY[:, -2] - pY[:, -1], base_indices, pY, previous_eeg_vector],
        'on_unused_input': 'ignore',
        'allow_input_downcast': True
    }

    params = chain.from_iterable(n.params for n in subnets)
    update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                         for param in params]
    basic_updates = []
    if combiner_on:
        basic_updates += [(
            skill_accumulator,
            T.set_subtensor(skill_accumulator[base_indices - 1], combiner_out)
        )]
    tf_valid = theano.function(
        updates=basic_updates,
        givens={t_dropout: 0.},
        **func_args)
    tf_train = theano.function(
        updates=update_parameters + basic_updates,
        givens={t_dropout: dropout_p},
        **func_args)

    def gen_batches(idxs, keys, batch_size=batch_size):
        all_keys = zip(*keys)
        full_batches = [list(islice(idx, 2, None)) for k, idx in
                        groupby(idxs, key=lambda i: all_keys[i])]

        def sub_batch(idxs):
            return [idxs[i * batch_size: (i + 1) * batch_size] for i in
                    xrange(int(len(idxs) / batch_size))]
        batches = list(chain.from_iterable(imap(sub_batch, full_batches)))
        return batches

    def f_eval(idxs, pred):
        _y = correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y[:len(pred)], pred, pos_label=1)

    def f_train(idxs, **kwargs):
        res = tf_train(idxs)
        return res[:3]

    def f_valid(idxs, **kwargs):
        res = tf_valid(idxs)
        return res[:3]

    train_batches = gen_batches(train_idx, [subject_x], batch_size)
    valid_batches = gen_batches(valid_idx, [subject_x], 1)
    return f_train, f_valid, train_batches, valid_batches, f_eval, f_eval
