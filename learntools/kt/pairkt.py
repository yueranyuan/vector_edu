from itertools import chain, imap, islice, ifilter

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams as TRandomStreams

from learntools.libs.utils import idx_to_mask, mask_to_idx
from learntools.libs.auc import auc
from learntools.model.theano_utils import make_shared
from learntools.model import gen_batches_by_keys, gen_batches_by_size
from learntools.kt.base import BaseKT

from theano import config

config.exception_verbosity = 'high'


def _gen_batches(idxs, subjects, batch_size):
    batches = gen_batches_by_keys(idxs, [subjects])
    batches = imap(lambda idxs: islice(idxs, 2, None), batches)
    sub_batches = imap(lambda idxs: gen_batches_by_size(list(idxs), batch_size=batch_size), batches)
    sub_batches_with_subj = map(lambda batch_group: [(batch, batch_group[0][0]) for batch in batch_group],
                                sub_batches)
    batches = chain.from_iterable(sub_batches_with_subj)
    batches = list(batches)
    batches = ifilter(lambda b: b[0], batches)  # remove empty batches
    batches = list(batches)
    return batches


class PairKT(BaseKT):
    '''a trainable, applyable model for logistic regression based kt with a previous result as reference
    Attributes:
        train_batches (int[][]): training batches are divided by subject_id and the rows of each subject
            are further divided by batch_size. The first 2 rows of each subject are removed by
            necessity due to the recursive structure of the model
        valid_batches (int[][]): validation batches. See train_batches
    '''
    def __init__(self, prepared_data, skill_matrix,
                 current_eeg_on=1, batch_size=30, previous_on=True, aggregate=0, day_on=True, time_vector_width=5, **kwargs):
        '''
        Args:
            prepared_data (tuple(Dataset, int[], int[])): a tuple that holds the data to be used,
                the row indices of the training set, and the row indices of the validation set
        '''
        self.aggregate = aggregate
        # ##########
        # STEP1: order the data properly so that we can read from it sequentially
        # when training the model

        ds, train_idx, valid_idx = prepared_data
        N = len(ds['correct'])
        eeg_vector_len = ds.get_data('eeg').shape[1]
        train_mask = idx_to_mask(train_idx, len(ds['subject']))
        valid_mask = idx_to_mask(valid_idx, len(ds['subject']))

        sorted_i = sorted(xrange(N), key=lambda(i): (ds['subject'][i]))
        ds.reorder(sorted_i)

        train_mask = train_mask[sorted_i]
        valid_mask = valid_mask[sorted_i]
        train_idx = mask_to_idx(train_mask)
        valid_idx = mask_to_idx(valid_mask)
        base_indices = T.ivector('idx')

        skill_x = ds.get_data('skill')
        subject_x = ds.get_data('subject')
        time_x = ds.get_data('start_time')
        correct_y = ds.get_data('correct') - 1
        eeg_full = ds.get_data('eeg')

        # ###########
        # STEP2: create the inputs for the model.

        # make a skill matrix containing skill vectors for each skill
        skill_matrix_width = skill_matrix.shape[1]
        skill_matrix = make_shared(skill_matrix, name='skill_matrix')

        # data preloaded into network
        skill_x = make_shared(skill_x, to_int=True, name='skill')
        if day_on:
            day_in_centiseconds = 100 * 60 * 60 * 24
            date_x = make_shared(time_x / day_in_centiseconds, to_int=True, name='date')
        correct_y = make_shared(correct_y, to_int=True, name='correct')
        eeg_full = make_shared(eeg_full, name='eeg')

        # create the classifier
        input_size = skill_matrix_width
        if previous_on:
            input_size += skill_matrix_width + 1
            if day_on:
                input_size += time_vector_width
        if current_eeg_on:
            input_size += eeg_vector_len
            if previous_on:
                input_size += eeg_vector_len

        # create components of the feature vector from previous rows
        if previous_on:
            # generate indices for a random previous skill exposure within this batch.
            prev_indices = T.ivector('prev_indices')
            subject_start_idx = base_indices[0]
            srng = TRandomStreams()
            rand_floats = srng.uniform(size=base_indices.shape)
            prev_indices_relative = T.cast((base_indices - subject_start_idx) * rand_floats, 'int32')
            prev_indices_absolute = prev_indices_relative + subject_start_idx

            # extract information from previous row
            previous_skill = skill_matrix[skill_x[prev_indices]]
            previous_eeg_vector = eeg_full[prev_indices]

            # get whether previous row was correct
            # need to convert list of indices of 1, 2 into [0],[1] columns
            correct_vectors = make_shared([[0], [1]])
            correct_feature = correct_vectors[correct_y[prev_indices]]
            correct_feature.name = 'correct_feature'

            # create date vector that represents how long ago the previous row was
            if day_on:
                # build a date vector table to index into
                days_back = np.zeros((2 ** time_vector_width, time_vector_width))
                for i in xrange(time_vector_width):
                    i_exp = 2 ** i
                    days_back[i_exp:(2 * i_exp), i] = 1
                farthest_day_back = days_back.shape[0] - 1
                t_days_back = make_shared(days_back, name='days_back')

                # compute which date vector to use for each index
                t_delta = T.cast(date_x[base_indices] - date_x[prev_indices], 'int64')
                t_delta.name = 't_delta'
                t_delta_capped = T.clip(t_delta, 0, farthest_day_back)
                t_delta_capped.name = 't_delta_capped'
                time_vector = t_days_back[t_delta_capped]

        current_skill = skill_matrix[skill_x[base_indices]]
        current_eeg_vector = eeg_full[base_indices]

        classifier_inputs = [current_skill]
        if previous_on:
            classifier_inputs += [correct_feature, previous_skill]
            if day_on:
                classifier_inputs += [time_vector]
        if current_eeg_on:
            classifier_inputs += [current_eeg_vector]
            if previous_on:
                classifier_inputs += [previous_eeg_vector]

        # ###########
        # STEP2.5: build the classifier
        y = correct_y[base_indices]
        self._correct_y = correct_y

        feature_vector = T.concatenate(classifier_inputs, axis=1)
        feature_vector.name = 'feature_vector'
        self.gen_inputs = theano.function(
            inputs=[base_indices, subject_start_idx],
            outputs=[feature_vector, y],
            givens=[(prev_indices, prev_indices_absolute)],
            allow_input_downcast=True)

        self.train_batches = _gen_batches(train_idx, subject_x, batch_size)
        self.valid_batches = _gen_batches(valid_idx, subject_x, batch_size)

    def _aggregated_validation(self, idxs, subject_idx):
        aggregate_length = 30
        aggregated_pY = np.zeros_like(idxs, dtype='float32')
        first_of_batch = min(idxs)
        for lookback in xrange(1, max(aggregate_length, first_of_batch - subject_idx)):
            res = self._tf_valid_lookback(idxs, subject_idx, lookback)
            loss, pY, out_idxs = res[:3]
            aggregated_pY += pY
        return loss, aggregated_pY, out_idxs

    def _get_data_from_batches(self, batches):
        first_loop = True

        for idxs, subject_start_idx in batches:
            feats, y = self.gen_inputs(idxs, subject_start_idx)

            if first_loop:
                all_feats = feats
                all_y = y
                first_loop = False
            else:
                all_feats = np.vstack((all_feats, feats))
                all_y = np.hstack((all_y, y))

        return all_feats, all_y

    def train_full(self, **kwargs):
        train_feats, train_y = self._get_data_from_batches(self.train_batches)

        print("training ...")
        self.classifier.fit(train_feats, train_y)
        print("finished training")

        sum_preds = None
        for i in xrange(20):
            test_feats, test_y = self._get_data_from_batches(self.valid_batches)
            preds = self.classifier.predict(test_feats)
            if sum_preds is None:
                sum_preds = preds
            else:
                sum_preds += preds
        print(sum_preds)
        print(auc(test_y, sum_preds, pos_label=1))