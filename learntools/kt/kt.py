from itertools import groupby

import numpy as np
import theano
import theano.tensor as T

from learntools.libs.logger import log_me
from learntools.libs.utils import idx_to_mask
from learntools.libs.auc import auc
from learntools.model.math import neg_log_loss
from learntools.model.theano_utils import make_shared, make_probability


@log_me('... building the model')
def build_model(prepared_data, clamp_L0=0.4, **kwargs):
    # ##########
    # STEP1: order the data properly so that we can read from it sequentially
    # when training the model

    subject_x, skill_x, correct_y, start_x, eeg_x, eeg_table, stim_pairs, train_idx, valid_idx = prepared_data
    N = len(correct_y)
    train_mask = idx_to_mask(train_idx, N)
    valid_mask = idx_to_mask(valid_idx, N)

    # sort data by subject and skill
    sorted_i = sorted(xrange(N), key=lambda i: (subject_x[i], skill_x[i], start_x[i]))
    skill_x = skill_x[sorted_i]
    subject_x = subject_x[sorted_i]
    correct_y = correct_y[sorted_i]
    start_x = start_x[sorted_i]
    train_mask = train_mask[sorted_i]
    valid_mask = valid_mask[sorted_i]
    train_idx = np.nonzero(train_mask)[0]
    valid_idx = np.nonzero(valid_mask)[0]

    n_skills = np.max(skill_x) + 1
    n_subjects = np.max(subject_x) + 1

    # prepare parameters
    p_T = 0.5
    p_G = 0.1
    p_S = 0.2
    if clamp_L0 is None:
        p_L0 = 0.7
    else:
        p_L0 = clamp_L0
    parameter_base = np.ones(n_skills)
    tp_L0, t_L0 = make_probability(parameter_base * p_L0, name='L0')
    tp_T, t_T = make_probability(parameter_base * p_T, name='p(T)')
    tp_G, t_G = make_probability(parameter_base * p_G, name='p(G)')
    tp_S, t_S = make_probability(parameter_base * p_S, name='p(S)')

    # declare and prepare variables for theano
    i = T.ivector('i')
    dummy_float = make_shared(0, name='dummy')
    skill_i, subject_i = T.iscalars('skill_i', 'subject_i')
    correct_y = make_shared(correct_y, to_int=True)

    def step(correct_i, prev_L, prev_p_C, P_T, P_S, P_G):
        Ln = prev_L + (1 - prev_L) * P_T
        p_C = prev_L * (1 - P_S) + (1 - prev_L) * P_G
        return Ln, p_C

    # set up theano functions
    ((results, p_C), updates) = theano.scan(fn=step,
                                            sequences=correct_y[i],
                                            outputs_info=[tp_L0[skill_i],
                                                          dummy_float],
                                            non_sequences=[tp_T[skill_i],
                                                           tp_G[skill_i],
                                                           tp_S[skill_i]])

    p_y = T.stack(1 - p_C, p_C)
    loss = neg_log_loss(p_y, correct_y[i])

    learning_rate = T.fscalar('learning_rate')
    if clamp_L0 is None:
        params = [t_T, t_L0]
    else:
        params = [t_T]
    update_parameters = [(param, param - learning_rate * T.grad(loss, param))
                         for param in params]

    tf_train = theano.function(inputs=[i, skill_i, learning_rate],
                               updates=update_parameters,
                               outputs=[loss, results, i],
                               allow_input_downcast=True)
    tf_valid = theano.function(inputs=[i, skill_i],
                               outputs=[loss, results, i],
                               allow_input_downcast=True)

    def f_train((i, (subject_i, skill_i)), learning_rate):
        return tf_train(i, skill_i, learning_rate)

    def f_valid((i, (subject_i, skill_i))):
        return tf_valid(i, skill_i)

    def gen_batches(idxs, keys):
        all_keys = zip(*keys)
        return [(list(idx), k) for k, idx in
                groupby(idxs, key=lambda i: all_keys[i])]

    def train_eval(idxs, pred):
        _y = correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y, pred, pos_label=1)

    def valid_eval(idxs, pred):
        _y = correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y, pred, pos_label=1)

    train_batches = gen_batches(train_idx, [subject_x, skill_x])
    valid_batches = gen_batches(valid_idx, [subject_x, skill_x])

    '''
    for (i, (subject_i, skill_i)) in train_batches:
        tf_train(i, skill_i, 0.1)
    for (i, (subject_i, skill_i)) in valid_batches:
        _1, _2, _3, a = tf_valid(i, skill_i)
        print sum(np.equal(a, 0.2)), max(skill_x) - len(np.unique(skill_x))

    import sys
    sys.exit()
    '''

    return f_train, f_valid, train_batches, valid_batches, train_eval, valid_eval
