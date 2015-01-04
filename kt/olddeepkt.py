import inspect
from itertools import compress, imap, chain, groupby, islice
from collections import namedtuple

import theano
import theano.tensor as T
import numpy as np

from model.vector import VectorLayer
from model.mlp import HiddenNetwork, MLP, rectifier
from libs.utils import normalize_table, make_shared, idx_to_mask
from libs.data import gen_word_matrix
from libs.logger import log, log_args
from libs.auc import auc


# look up tables are cheaper memory-wise.
# TODO: unify this implementation with VectorLayer
def to_lookup_table(x, access_idxs):
    mask = np.asarray([v is not None for v in x], dtype=bool)
    if not mask.any():
        raise Exception("can't create lookup table from no data")

    # create lookup table
    valid_idxs = np.nonzero(mask)[0]
    width = len(x[valid_idxs[0]])
    table = np.zeros((1 + len(valid_idxs), width))  # leave the first row for "None"
    for i, l in enumerate(compress(x, mask)):
        table[i + 1] = np.asarray(l)
    table[1:] = normalize_table(table[1:])
    table[0] = table[1:].mean(axis=0)  # set the "None" vector to the average of all vectors

    # create a way to index into lookup table
    idxs = np.zeros(len(x))
    idxs[valid_idxs] = xrange(1, len(valid_idxs) + 1)

    # convert to theano
    t_table = make_shared(table)
    t_idxs = make_shared(idxs, to_int=True)
    return t_table[t_idxs[access_idxs]], table.shape


def build_model(prepared_data, L1_reg, L2_reg, dropout_p, learning_rate,
                skill_vector_len=100, combiner_depth=1, combiner_width=200,
                main_net_depth=1, main_net_width=500, previous_eeg_on=1,
                current_eeg_on=1, mutable_skill=1, valid_percentage=0.8, batch_size=30,
                **kwargs):
    log('... building the model', True)
    log_args(inspect.currentframe())

    # ##########
    # STEP1: order the data properly so that we can read from it sequentially
    # when training the model

    subject_x, skill_x, correct_y, start_x, eeg_x, stim_pairs, train_idx, valid_idx = prepared_data
    correct_y += 1
    subject_x = subject_x[:, None]  # add extra dimension as a 'feature vector'
    skill_x = skill_x[:, None]  # add extra dimension as a 'feature vector'
    train_mask = idx_to_mask(train_idx, len(subject_x))
    valid_mask = idx_to_mask(valid_idx, len(subject_x))

    # reorder indices so that each index can be fed as a 'base_index' into the
    # full model. This means lining up by subjects and removing the first few indices.
    # first we sort by subject
    sorted_i = [i for (i, v) in
                sorted(enumerate(subject_x), key=lambda (i, v): v)]
    skill_x = skill_x[sorted_i]
    subject_x = subject_x[sorted_i]
    correct_y = correct_y[sorted_i]
    train_mask = train_mask[sorted_i]
    valid_mask = valid_mask[sorted_i]
    # then we get rid of the first few indices per subject
    subject_groups = groupby(range(len(subject_x)), lambda (i): subject_x[i, 0])
    good_indices = list(chain.from_iterable(
        imap(lambda (_, g): islice(g, 2, None), subject_groups)))

    t_good_indices = make_shared(good_indices, to_int=True)
    master_indices = T.ivector('idx')
    base_indices = t_good_indices[master_indices]

    # select only the good indices from CV indices
    good_mask = idx_to_mask(good_indices, len(subject_x))
    train_idx = np.nonzero(good_mask & train_mask)[0]
    valid_idx = np.nonzero(good_mask & valid_mask)[0]

    log('training set size: {}'.format(len(train_idx)))
    log('validation set size: {}'.format(len(valid_idx)))

    skill_x = make_shared(skill_x)
    subject_x = make_shared(subject_x)
    correct_y = make_shared(correct_y, to_int=True)

    # ###########
    # STEP2: connect up the model. See figures/vector_edu_model.png for diagram
    # TODO: make the above mentioned diagram

    NetInput = namedtuple("NetInput", ['input', 'size'])
    rng = np.random.RandomState(1234)
    t_dropout = T.scalar('dropout')
    y = correct_y[base_indices]

    # setup base-input layers
    skill_matrix = np.asarray(gen_word_matrix(skill_x.get_value(borrow=True),
                                              stim_pairs,
                                              vector_length=skill_vector_len),
                              dtype=theano.config.floatX)
    current_skill = VectorLayer(rng=rng,
                                indices=base_indices,
                                full_input=skill_x,
                                vectors=skill_matrix,
                                mutable=mutable_skill)
    previous_skill = VectorLayer(rng=rng,
                                 indices=base_indices - 1,
                                 full_input=skill_x,
                                 vectors=skill_matrix,
                                 mutable=mutable_skill)

    # setup combiner component
    skill_accumulator = make_shared(np.zeros(
        (skill_x.get_value(borrow=True).shape[0], combiner_width)))
    correct_feature = make_shared([[0], [1]])[correct_y[base_indices - 1] - 1]
    combiner_inputs = [NetInput(skill_accumulator[base_indices - 2], combiner_width),
                       NetInput(previous_skill.output, skill_vector_len),
                       NetInput(correct_feature, 1)]
    if previous_eeg_on:
        eeg_vector, (_, eeg_vector_len) = to_lookup_table(eeg_x, base_indices - 1)
        combiner_inputs.append(NetInput(eeg_vector, eeg_vector_len))
    combiner = HiddenNetwork(
        rng=rng,
        input=T.concatenate([c.input for c in combiner_inputs], axis=1),
        n_in=sum(c.size for c in combiner_inputs),
        size=[combiner_width] * combiner_depth,
        activation=rectifier,
        dropout=t_dropout
    )

    # setup main network component
    classifier_inputs = [NetInput(combiner.output, combiner_width),
                         NetInput(current_skill.output, skill_vector_len)]
    if current_eeg_on:
        eeg_vector, (_, eeg_vector_len) = to_lookup_table(eeg_x, base_indices)
        classifier_inputs.append(NetInput(eeg_vector, eeg_vector_len))
    classifier = MLP(rng=rng,
                     n_in=sum(c.size for c in classifier_inputs),
                     input=T.concatenate([c.input for c in classifier_inputs], axis=1),
                     size=[main_net_width] * main_net_depth,
                     n_out=3,
                     dropout=t_dropout)

    # ########
    # STEP3: create the theano functions to run the model

    subnets = (combiner, classifier)
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * sum([n.L1 for n in subnets])
        + L2_reg * sum([n.L2_sqr for n in subnets])
    )

    func_args = {
        'inputs': [base_indices],
        'outputs': [classifier.errors(y), classifier.output, base_indices],
        'on_unused_input': 'ignore',
        'allow_input_downcast': True
    }

    params = chain(current_skill.params, *[n.params for n in subnets])
    update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                         for param in params]
    update_accumulator = [(
        skill_accumulator,
        T.set_subtensor(skill_accumulator[base_indices - 1], combiner.output)
    )]
    tf_valid = theano.function(
        updates=update_accumulator,
        givens={t_dropout: 0.},
        **func_args)
    tf_train = theano.function(
        updates=update_parameters + update_accumulator,
        givens={t_dropout: dropout_p},
        **func_args)

    def gen_batches(idxs, batch_size):
        return [idxs[i * batch_size: (i + 1) * batch_size] for i in
                xrange(int(len(idxs) / batch_size))]

    def f_eval(idxs, pred):
        _y = correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y[:len(pred)], pred, pos_label=2)

    def f_train(idxs, **kwargs):
        return tf_train(idxs)

    def f_valid(idxs, **kwargs):
        return tf_valid(idxs)

    train_batches = gen_batches(train_idx, batch_size)
    valid_batches = gen_batches(valid_idx, 1)
    return f_train, f_valid, train_batches, valid_batches, f_eval, f_eval
