from itertools import imap, chain, groupby, islice, ifilter

import theano
import theano.tensor as T

from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.libs.logger import log_me
from learntools.model.logistic import LogisticRegression
from learntools.model.theano_utils import make_shared
from learntools.kt.deepkt import deep_gen_batches
from learntools.kt.base import BaseKT

from theano import config

config.exception_verbosity = 'high'


class LogKT(BaseKT):
    '''a trainable, applyable model for logistic regression based kt
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    @log_me('...building logkt')
    def __init__(self, prepared_data, skill_matrix, L1_reg=0., L2_reg=0., learning_rate=0.02,
                 current_eeg_on=1, batch_size=30, **kwargs):
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
        correct_y = ds.get_data('correct') - 1
        eeg_full = ds.get_data('eeg')

        # ###########
        # STEP2: connect up the model.

        # make a skill matrix containing skill vectors for each skill
        skill_matrix_width = skill_matrix.shape[1]
        skill_matrix = make_shared(skill_matrix, name='skill_matrix')

        # data preloaded into network
        skill_x = make_shared(skill_x, to_int=True, name='skill')
        correct_y = make_shared(correct_y, to_int=True, name='correct')
        eeg_full = make_shared(eeg_full, name='eeg')

        # rng = np.random.RandomState(1234)

        input_size = skill_matrix_width
        if current_eeg_on:
            input_size += eeg_vector_len
        classifier = LogisticRegression(n_in=input_size,
                                        n_out=2)

        # STEP 3.1 stuff that goes in scan
        current_skill = skill_matrix[skill_x[base_indices]]
        current_eeg_vector = eeg_full[base_indices]

        classifier_inputs = [current_skill]
        if current_eeg_on:
            classifier_inputs.append(current_eeg_vector)
        # probability of y for each 0, 1, 2
        pY = classifier.instance(T.concatenate(classifier_inputs, axis=1))

        # ########
        # STEP3: create the theano functions to run the model

        y = correct_y[base_indices]
        loss = -T.mean(T.log(pY)[T.arange(y.shape[0]), y])
        subnets = [classifier]
        # used to help compute regularization terms
        cost = (
            loss
            + L1_reg * sum([n.L1 for n in subnets])
            + L2_reg * sum([n.L2_sqr for n in subnets])
        )

        # the same for both validation and training
        func_args = {
            'inputs': [base_indices],
            'outputs': [loss, pY[:, 1] - pY[:, 0], base_indices],
            'on_unused_input': 'ignore',
            'allow_input_downcast': True,
        }

        # collect all theano updates
        params = chain.from_iterable(n.params for n in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params]
        # validation uses no dropout and previous skill propagation
        self._tf_valid = theano.function(
            **func_args)
        # training uses parameter updates plus previous skill propagation with dropout
        self._tf_train = theano.function(
            updates=update_parameters,
            **func_args)
        self.train_batches = deep_gen_batches(train_idx, subject_x, batch_size)
        self.valid_batches = deep_gen_batches(valid_idx, subject_x, 1)
        self._correct_y = correct_y