from itertools import groupby

import numpy as np
import theano
import theano.tensor as T

from learntools.libs.logger import log_me
from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.data import gen_word_matrix
from learntools.libs.auc import auc
from learntools.model.math import neg_log_loss, sigmoid
from learntools.model.theano_utils import make_shared, make_probability

def _gen_batches(idxs, subjects, batch_size):
    batches = gen_batches_by_keys(idxs, [subjects])
    batches = imap(lambda idxs: islice(idxs, 2, None), batches)
    sub_batches = imap(lambda idxs: gen_batches_by_size(list(idxs), batch_size), batches)
    batches = chain.from_iterable(sub_batches)
    batches = ifilter(lambda b: b, batches)
    batches = list(batches)
    return batches


class BLRmodel:
    @log_me('... building the model')
    def __init__(self, prepared_data, clamp_L0=None, skill_vector_len=100, **kwargs):
    # ##########
    # STEP1: order the data properly so that we can read from it sequentially
    # when training the model

        #subject_x, skill_x, correct_y, start_x, eeg_x, eeg_table, stim_pairs, train_idx, valid_idx = prepared_data
        
        ds, train_idx, valid_idx = prepared_data
        N = len(ds['correct'])
        #eeg_vector_len = ds.get_data('eeg').shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['subject']))
        valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

        sorted_i = sorted(xrange(N), key=lambda i: (ds['subject'][i]))
        ds.reorder(sorted_i)
        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        train_idx = mask_to_idx(train_mask)
        valid_idx = mask_to_idx(valid_mask)
        base_indices = T.ivector('idx')
        eeg_vector_len = ds.get_data('eeg').shape[1]
        skill_x = ds.get_data('skill')
        subject_x = ds.get_data('subject')
        correct_y = ds.get_data('correct')
        eeg_full = ds.get_data('eeg')
        start_x = ds.get_data('start_time')
        
       # sort data by subject and skill
        sorted_i = sorted(xrange(N), key=lambda i: (subject_x[i], skill_x[i], start_x[i]))
        skill_x = skill_x[sorted_i]
        start_x = start_x[sorted_i]
        subject_x = subject_x[sorted_i]
        correct_y = correct_y[sorted_i]
        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        train_idx = np.nonzero(train_mask)[0]
        valid_idx = np.nonzero(valid_mask)[0]
        skill_matrix = make_shared(gen_word_matrix(ds.get_data('skill'),
                                                   ds['skill'].enum_pairs,
                                                   vector_length=skill_vector_len))

        skill_x = make_shared(skill_x, to_int=True)
        correct_y = make_shared(correct_y, to_int=True)
        eeg_full = make_shared(eeg_full)
        n_skills = T.max(skill_x) + 1
        #print "hell%d" % n_skills
        # ####
        # STEP 2: initialize parameters
        p_G = 0.1
        p_S = 0.2
        feat_full = eeg_full
        #k = T.matrix('feat_full')
        #feat_columns = theano.function([k], (k ** 1).shape)
        #feat_table = eeg_x
        feat_columns = T.arange(feat_full.shape[1])  # [0, 1, 2, 3, 4, 5, 6]

        feat_width = T.arange(feat_full.shape[0])
        if clamp_L0 is None:
            Beta0 = make_shared(np.random.randn(n_skills, to_int=True)
        Beta = make_shared(np.random.rand(n_skills, feat_width))
        b = make_shared(np.random.randn(n_skills))
        Gamma = make_shared(np.random.randn(n_skills, feat_width))
        g = make_shared(np.random.randn(n_skills))
        tp_G, t_G = make_probability(p_G, name='p(G)')
        tp_S, t_S = make_probability(p_S, name='p(S)')

        # declare and prepare variables for theano
        i = T.ivector('i')
        dummy_float = make_shared(0, name='dummy')
        skill_i, subject_i = T.iscalars('skill_i', 'subject_i')
        correct_y = make_shared(correct_y, to_int=True)
        feat_full = make_shared(feat_full, to_int=True)
        #feat_table = make_shared(feat_table)

        # set up theano functions
        def step(correct_i, feat, prev_L, prev_p_C, skill_i, P_S, P_G):
            L_true_given_true = sigmoid(T.dot(Beta[skill_i].T, feat[feat_columns]) + b[skill_i])
            L_true_given_false = sigmoid(T.dot(Gamma[skill_i].T, feat[feat_columns]) + g[skill_i])
            Ln = prev_L * L_true_given_true + (1 - prev_L) * L_true_given_false
            p_C = prev_L * (1 - P_S) + (1 - prev_L) * P_G
            return Ln, p_C
        if clamp_L0 is None:
            L0 = sigmoid(Beta0[skill_i])
        else:
            L0 = make_shared(clamp_L0)
        ((results, p_C), updates) = theano.scan(fn=step,
                                                sequences=[correct_y[i],
                                                           feat_full[feat_full[i]]],
                                                outputs_info=[L0,
                                                              dummy_float],
                                                non_sequences=[skill_i,
                                                               tp_G,
                                                               tp_S])
        p_y = T.stack(1 - p_C, p_C)
        loss = neg_log_loss(p_y, correct_y[i])

        learning_rate = T.fscalar('learning_rate')
        if clamp_L0 is None:
            params = [Beta0, Beta, Gamma, g, b]
        else:
            params = [Beta, Gamma, g, b]
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

        train_batches = _gen_batches(train_idx, [subject_x, skill_x])
        valid_batches = _gen_batches(valid_idx, [subject_x, skill_x])

        return f_train, f_valid, train_batches, valid_batches, train_eval, valid_eval
