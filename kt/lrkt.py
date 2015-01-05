import inspect
from itertools import groupby

import numpy as np
import theano
import theano.tensor as T

from libs.logger import log, log_args
from libs.utils import idx_to_mask, make_shared
from libs.auc import auc
from libs.data import gen_word_matrix


def make_probability(init, shape=None, **kwargs):
    if shape:
        init = np.ones(shape) * init
    logit_p = np.log(init / (1 - init))
    logit_p = make_shared(logit_p, **kwargs)
    return 1 / (1 + T.exp(-logit_p)), logit_p


def sigmoid(x):
    return 1 / (1 + T.exp(-x))


def neg_log_loss(p, y):
    return -T.sum(T.log(p.T)[T.arange(y.shape[0]), y])


def build_model(prepared_data, **kwargs):
    log('... building the model', True)
    log_args(inspect.currentframe())

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

    skill_table = np.diag(np.ones(n_skills))
    # skill_table = gen_word_matrix(skill_x, stim_pairs, vector_length=100)
    skill_table_width = skill_table.shape[1]

    # ####
    # STEP 2: initialize parameters
    p_G = 0.1
    p_S = 0.2
    eeg_columns = [0, 1, 2, 3]
    eeg_width = len(eeg_columns)
    Beta0 = make_shared(np.random.rand(skill_table_width).T)
    Beta = make_shared(np.random.rand(skill_table_width + eeg_width).T)
    Gamma = make_shared(np.random.rand(skill_table_width + eeg_width).T)
    tp_G, t_G = make_probability(p_G, name='p(G)')
    tp_S, t_S = make_probability(p_S, name='p(S)')

    # declare and prepare variables for theano
    i = T.ivector('i')
    dummy_float = make_shared(0, name='dummy')
    skill_i, subject_i = T.iscalars('skill_i', 'subject_i')
    correct_y = make_shared(correct_y, to_int=True)
    skill_table = make_shared(skill_table, to_int=True)
    # eegs = make_shared(eeg_table[np.asarray(eeg_x, dtype='int32')])
    eeg_x = make_shared(eeg_x, to_int=True)
    eeg_table = make_shared(eeg_table)

    def step2(eeg):
        return eeg

    # set up theano functions
    def step(correct_i, eeg, prev_L, prev_p_C, subskills, P_S, P_G):
        x = T.concatenate((subskills, eeg[eeg_columns]))
        L_true_given_true = sigmoid(T.dot(Beta, x))
        L_true_given_false = sigmoid(T.dot(Gamma, x))
        Ln = prev_L * L_true_given_true + (1 - prev_L) * L_true_given_false
        p_C = prev_L * (1 - P_S) + (1 - prev_L) * P_G
        return Ln, p_C
    L0 = sigmoid(T.dot(Beta0, skill_table[skill_i]))
    ((results, p_C), updates) = theano.scan(fn=step,
                                            sequences=[correct_y[i],
                                                       eeg_table[eeg_x[i]]],
                                            outputs_info=[L0,
                                                          dummy_float],
                                            non_sequences=[skill_table[skill_i],
                                                           tp_G,
                                                           tp_S])
    p_y = T.stack(1 - p_C, p_C)
    loss = neg_log_loss(p_y, correct_y[i])

    learning_rate = T.fscalar('learning_rate')
    params = [Beta0, Beta, Gamma]
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
        everything = tf_train(i, skill_i, learning_rate)
        return everything[:3]

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

    return f_train, f_valid, train_batches, valid_batches, train_eval, valid_eval
