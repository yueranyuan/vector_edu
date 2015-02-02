from itertools import groupby

import numpy as np
import theano
import theano.tensor as T

from learntools.libs.logger import log_me
from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.data import gen_word_matrix
from learntools.libs.auc import auc
from learntools.model.math import neg_log_loss
from learntools.model.theano_utils import make_shared, make_probability
from learntools.model import Model, gen_batches_by_keys, gen_batches_by_size
from itertools import imap, chain, groupby, islice, ifilter

def _gen_batches(idxs, subjects, batch_size):
    batches = gen_batches_by_keys(idxs, [subjects])
    batches = imap(lambda idxs: islice(idxs, 2, None), batches)
    sub_batches = imap(lambda idxs: gen_batches_by_size(list(idxs), batch_size), batches)
    batches = chain.from_iterable(sub_batches)
    batches = ifilter(lambda b: b, batches)
    batches = list(batches)
    return batches

class Bmodel(Model):
    # ##########
    # STEP1: order the data properly so that we can read from it sequentially
    # when training the model
    @log_me('... building the model')
    def __init__(self, prepared_data, skill_vector_len=100, clamp_L0=0.4, batch_size=30, eeg_column_i=None, **kwargs):
        
        ds, train_idx, valid_idx = prepared_data
        N = len(ds['correct'])
        eeg_vector_len = ds.get_data('eeg').shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['subject']))
        valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

        sorted_i = sorted(xrange(N), key=lambda i: (ds['subject'][i], ds['skill'][i], ds['start_time'][i]))
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

        n_skills = np.max(skill_x) +1
        skill_x = make_shared(skill_x, to_int=True)
        correct_y = make_shared(correct_y, to_int=True)
        n_subjects = np.max(subject_x) + 1

    # binarize eeg
        eeg_single_x = np.zeros(N)
        if eeg_column_i is not None:
            eeg_column = eeg_table[eeg_full, eeg_column_i]
            above_median = np.greater(eeg_column, np.median(eeg_column))
            eeg_single_x[above_median] = 1

    # prepare parameters
        p_T = 0.5
        p_G = 0.1
        p_S = 0.2
        p_L0 = 0.7
        if clamp_L0 is None:
           p_L0 = 0.7
        else:
          p_L0 = clamp_L0
        parameter_base = np.ones(n_skills)
        tp_L0, t_L0 = make_probability(parameter_base * p_L0, name='L0')
        tp_T, t_T = make_probability(np.ones((n_skills, 2)) * p_T, name='p(T)')
        tp_G, t_G = make_probability(p_G, name='p(G)')
        tp_S, t_S = make_probability(p_S, name='p(S)')

    # declare and prepare variables for theano
        i = T.ivector('i')
        dummy_float = make_shared(0, name='dummy')
        skill_i, subject_i = T.iscalars('skill_i', 'subject_i')
        #correct_y = make_shared(correct_y, to_int=True)
        eeg_single_x = make_shared(eeg_single_x, to_int=True)

    def step(correct_i, eeg, prev_L, prev_p_C, P_T, P_S, P_G):
        Ln = prev_L + (1 - prev_L) * P_T[eeg]
        p_C = prev_L * (1 - P_S) + (1 - prev_L) * P_G
        return Ln, p_C

    # set up theano functions
        ((results, p_C), updates) = theano.scan(fn=step,
                                                sequences=[correct_y[i],
                                                           eeg_single_x[i]],
                                                outputs_info=[tp_L0[skill_i],
                                                              dummy_float],
                                                non_sequences=[tp_T[skill_i],
                                                               tp_G,
                                                               tp_S])
 
        p_y = T.stack(1 - p_C, p_C)
        loss = neg_log_loss(p_y, correct_y[i])
 
        learning_rate = T.fscalar('learning_rate')
        if clamp_L0 is None:
            params = [t_T, t_L0]
        else:
            params = [t_T]
        update_parameters = [(param, param - learning_rate * T.grad(loss, param))
                             for param in params]
 
        self._tf_train = theano.function(inputs=[i, skill_i, learning_rate],
                                   updates=update_parameters,
                                   outputs=[loss, results, i],
                                   allow_input_downcast=True)
        self._tf_valid = theano.function(inputs=[i, skill_i],
                                   outputs=[loss, results, i],
                                   allow_input_downcast=True)

        
        self.train_batches = _gen_batches(train_idx, subject_x, batch_size)
        self.valid_batches = _gen_batches(valid_idx, subject_x, 1)
        self._correct_y = correct_y
 
    #def f_train((i, (subject_i, skill_i)), learning_rate):
    #    return _tf_train(i, skill_i, learning_rate)
 
    #def f_valid((i, (subject_i, skill_i))):
    #    return _tf_valid(i, skill_i)

        
    def evaluate(self, idxs, pred):
        _y = self._correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y[:len(pred)], pred, pos_label=1)

    def train(self, **kwargs):
        res = self._tf_train(i, skill_i, learning_rate)
        return res[:3]

    def validate(self, idxs, **kwargs):
        res = self._tf_valid(i, skill_i)
        return res[:3]


        '''
        for (i, (subject_i, skill_i)) in train_batches:
            tf_train(i, skill_i, 0.1)
        for (i, (subject_i, skill_i)) in valid_batches:
            _1, _2, _3, a = tf_valid(i, skill_i)
            print sum(np.equal(a, 0.2)), max(skill_x) - len(np.unique(skill_x))

        import sys
        sys.exit()
        '''

  