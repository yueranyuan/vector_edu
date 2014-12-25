import sys
import time
import inspect
import cPickle
import gzip
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy
import theano
import theano.tensor as T

from model.mlp import MLP, HiddenLayer, rectifier
from model.vector import VectorLayer
from libs.utils import gen_log_name, make_shared, random_unique_subset
import config
from data import gen_word_matrix
from itertools import imap, islice, groupby, chain
from libs.auc import auc


def get_val(tensor):
    return tensor.get_value(borrow=True)


def log(txt, also_print=False):
    if also_print:
        print txt
    with open(LOG_FILE, 'a+') as f:
        f.write('{0}\n'.format(txt))


def build_model(prepared_data, L1_reg, L2_reg, n_hidden, dropout_p,
                learning_rate):
    log('... building the model', True)
    subject_x, skill_x, correct_y, eeg_x, stim_pairs = prepared_data
    subject_x = subject_x[:, None]  # add extra dimension as a 'feature vector'
    skill_x = skill_x[:, None]  # add extra dimension as a 'feature vector'

    # reorder indices so that each index can be fed as a 'base_index' into the
    # full model. This means lining up by subjects and removing the first few indices.
    # first we sort by subject
    sorted_i = [i for (i, v) in
        sorted(enumerate(subject_x), key=lambda (i, v): v)]
    skill_x = skill_x[sorted_i]
    subject_x = subject_x[sorted_i]
    correct_y = correct_y[sorted_i]
    # then we get rid of the first few indices per subject
    subject_groups = groupby(range(len(subject_x)), lambda (i): subject_x[i, 0])
    good_indices = list(chain.from_iterable(
        imap(lambda (_, g): islice(g, 2, None), subject_groups)))

    t_good_indicies = make_shared(good_indices, to_int=True)
    master_indices = T.ivector('idx')
    base_indices = t_good_indicies[master_indices]

    # create cv folds
    #validation = (random_unique_subset(subject_x[good_indices, 0]) &
    #              random_unique_subset(skill_x[good_indices, 0]))
    validation = random_unique_subset(subject_x[good_indices, 0])
    train_idx = numpy.nonzero(numpy.logical_not(validation))[0]
    valid_idx = numpy.nonzero(validation)[0]

    skill_x = make_shared(skill_x)
    subject_x = make_shared(subject_x)
    correct_y = make_shared(correct_y, to_int=True)

    # setup the layers
    rng = numpy.random.RandomState(1234)
    skill_vector_len = 100
    t_dropout = T.scalar('dropout')
    y = correct_y[base_indices]

    skill_matrix = numpy.asarray(gen_word_matrix(skill_x.get_value(borrow=True),
                                                 stim_pairs,
                                                 vector_length=skill_vector_len),
                                 dtype=theano.config.floatX)
    skill_vectors = VectorLayer(rng=rng,
                                indices=base_indices,
                                full_input=skill_x,
                                vectors=skill_matrix)
    skill_vectors1 = VectorLayer(rng=rng,
                                 indices=base_indices - 1,
                                 full_input=skill_x,
                                 vectors=skill_matrix)
    # skill_vectors2 = VectorLayer(rng=rng,
    #                             indices=base_indices - 2,
    #                             full_input=skill_x,
    #                             vectors=skill_matrix)
    '''
    subject_vector_len = 50
    subject_vectors = VectorLayer(rng=rng,
                                  indices=base_indices,
                                  full_input=subject_x,
                                  n_skills=max(subject_x.get_value(borrow=True)) + 1,
                                  vector_length=subject_vector_len,
                                  mutable=False)
    '''
    knowledge_vector_len = 100
    skill_accumulator = make_shared(numpy.zeros(
        (skill_x.get_value(borrow=True).shape[0], knowledge_vector_len)))
    combiner = HiddenLayer(
        rng=rng,
        input=T.concatenate([skill_accumulator[base_indices - 2], skill_vectors1.output], axis=1),
        n_in=skill_vector_len + knowledge_vector_len,
        n_out=knowledge_vector_len,
        activation=rectifier,
        dropout=t_dropout
    )
    classifier = MLP(rng=rng,
                     n_in=knowledge_vector_len + skill_vector_len,
                     input=T.concatenate([combiner.output, skill_vectors.output], axis=1),
                     n_hidden=n_hidden,
                     n_out=3,
                     dropout=t_dropout)
    subnets = (combiner, classifier)
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * sum([n.L1 for n in subnets])
        + L2_reg * sum([n.L2_sqr for n in subnets])
    )

    func_args = {
        'inputs': [base_indices],
        'outputs': [classifier.errors(y), classifier.output],
        'givens': {t_dropout: dropout_p},
        'on_unused_input': 'ignore',
        'allow_input_downcast': True
    }

    params = classifier.params + skill_vectors.params + combiner.params
    update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                         for param in params]
    update_accumulator = [(skill_accumulator,
        T.set_subtensor(skill_accumulator[base_indices - 1], combiner.output))]
    f_valid = theano.function(updates=update_accumulator, **func_args)
    f_train = theano.function(updates=update_parameters + update_accumulator, **func_args)

    def validator_func(pred):
        _y = correct_y.owner.inputs[0].get_value(borrow=True)[valid_idx]
        return auc(_y[:len(pred)], pred, pos_label=2)
    return f_train, f_valid, train_idx, valid_idx, validator_func


def train_model(train_model, validate_model, train_idx, valid_idx, validator_func,
                batch_size, n_epochs):
    log('... training', True)

    patience = 100000  # look as this many examples regardless
    patience_increase = 2
    improvement_threshold = 0.998
    validation_frequency = 5
    best_valid_error = numpy.inf
    best_iter = 0
    iteration = 0

    for epoch in range(n_epochs):
        # before the skill_accumulator is setup properly, train one at a time
        _batch_size = 1 if epoch == 0 else batch_size
        for minibatch_index in xrange(int(len(train_idx) / _batch_size)):
            train_model(train_idx[minibatch_index * _batch_size: (minibatch_index + 1) * _batch_size])
            iteration = iteration + 1

        if (epoch + 1) % validation_frequency == 0:
            _batch_size = 1  # recreate the skill_accumulator each time
            results = [validate_model(valid_idx[i * _batch_size: (i + 1) * _batch_size])
                       for i in xrange(int(len(train_idx) / _batch_size))]
            # Aaron: this is not really a speed critical part of the code but we
            # can come back and redo AUC in theano if we want to make this suck less
            predictions = list(chain.from_iterable(imap(lambda r: r[1], results)))
            valid_error = validator_func(predictions)
            log('epoch {epoch}, validation error {err:.2%}'.format(
                epoch=epoch, err=valid_error))

            if valid_error < best_valid_error:
                if (valid_error < best_valid_error * improvement_threshold):
                    patience = max(patience, iteration * patience_increase)
                best_valid_error = valid_error
                best_iter = iteration

            if patience <= iteration:
                break
    return best_valid_error, best_iter, iteration


def run(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
        dataset_name='data/data.gz', batch_size=30, n_hidden=500, dropout_p=0.2):
    args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    arg_summary = ', '.join(['{0}={1}'.format(arg, eval(arg)) for arg in args])
    log(arg_summary)

    with gzip.open(dataset_name, 'rb') as f:
        prepared_data = cPickle.load(f)

    f_train, f_validate, train_idx, valid_idx, validator_func = (
        build_model(prepared_data, L1_reg=L1_reg, L2_reg=L2_reg,
                    n_hidden=n_hidden, dropout_p=dropout_p,
                    learning_rate=learning_rate))

    start_time = time.clock()
    best_validation_loss, best_iter, iteration = (
        train_model(f_train, f_validate, train_idx, valid_idx, validator_func,
                    batch_size, n_epochs=n_epochs))
    end_time = time.clock()
    training_time = (end_time - start_time) / 60.

    log(('Optimization complete. Best validation score of %f %% '
         'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, 0.), True)
    log('Code ran for ran for %.2fm' % (training_time))
    return (best_validation_loss * 100., best_iter + 1, iteration, training_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run a theano experiment on this computer")
    parser.add_argument('-p', dest='param_set', type=str, default='default',
                        choices=config.all_param_set_keys,
                        help='the name of the parameter set that we want to use')
    parser.add_argument('--f', dest='file', type=str, default='data/data.gz',
                        help='the data file to use')
    parser.add_argument('-o', dest='outname', type=str, default=gen_log_name(),
                        help='name for the log file to be generated')
    args = parser.parse_args()

    params = config.get_config(args.param_set)
    LOG_FILE = args.outname
    log(run(dataset_name=args.file, **params))
    print "finished"
    if sys.platform.startswith('win'):
        from win_utils import winalert
        winalert()
