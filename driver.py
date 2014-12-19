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
from itertools import islice, groupby, chain


def get_val(tensor):
    return tensor.get_value(borrow=True)


def log(txt, also_print=False):
    if also_print:
        print txt
    with open(LOG_FILE, 'a+') as f:
        f.write('{0}\n'.format(txt))


def prepare_data(dataset_name, batch_size):
    log('... loading data', True)

    with gzip.open(dataset_name, 'rb') as f:
        dataset = cPickle.load(f)
    skill_x, subject_x, correct_y, stim_pairs = dataset
    return (skill_x, subject_x, correct_y, stim_pairs)


def build_model(prepared_data, L1_reg, L2_reg, n_hidden, dropout_p,
                learning_rate):
    log('... building the model', True)
    skill_x, subject_x, correct_y, stim_pairs = prepared_data

    # reorder indices so that each index can be fed as a 'base_index' into the
    # full model. This means lining up by subjects and removing the first few indices.
    # first we sort by subject
    sorted_i = [i for (i, v) in
        sorted(enumerate(subject_x), key=lambda (i, v): v)]
    skill_x = skill_x[sorted_i]
    subject_x = subject_x[sorted_i]
    correct_y = correct_y[sorted_i]
    # then we get rid of the first few indices per subject
    valid_indices = list(chain.from_iterable(
        list(islice(g, 2, None)) for _, g in
        groupby(range(len(subject_x)), lambda (i): subject_x[i, 0])))

    t_valid_indicies = make_shared(valid_indices, to_int=True)
    master_indices = T.ivector('idx')
    base_indices = t_valid_indicies[master_indices]

    # create cv folds
    validation = (random_unique_subset(subject_x[valid_indices, 0]) &
                  random_unique_subset(skill_x[valid_indices, 0]))
    train_idx = numpy.nonzero(numpy.logical_not(validation))[0]
    valid_idx = numpy.nonzero(validation)[0]

    skill_x = make_shared(skill_x)
    subject_x = make_shared(subject_x)
    correct_y = make_shared(correct_y, to_int=True)

    rng = numpy.random.RandomState(1234)
    skill_vector_len = 50
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
    skill_vectors2 = VectorLayer(rng=rng,
                                 indices=base_indices - 2,
                                 full_input=skill_x,
                                 vectors=skill_matrix)
    '''
    subject_vector_len = 50
    subject_vectors = VectorLayer(rng=rng,
                                  indices=base_indices,
                                  full_input=subject_x,
                                  n_skills=max(subject_x.get_value(borrow=True)) + 1,
                                  vector_length=subject_vector_len,
                                  mutable=False)
    '''
    combiner = HiddenLayer(
        rng=rng,
        input=T.concatenate([skill_vectors1.output, skill_vectors2.output], axis=1),
        n_in=2 * skill_vector_len,
        n_out=n_hidden,
        activation=rectifier,
        dropout=t_dropout
    )
    classifier = MLP(rng=rng,
                     n_in=n_hidden + skill_vector_len,
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
        'outputs': [classifier.errors(y)],
        'givens': {t_dropout: dropout_p},
        'on_unused_input': 'ignore',
        'allow_input_downcast': True
    }

    params = classifier.params + skill_vectors.params + combiner.params
    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in params]
    f_valid = theano.function(**func_args)
    f_train = theano.function(updates=updates, **func_args)
    return f_train, f_valid, train_idx, valid_idx


def train_model(train_model, validate_model,
                batch_size, train_idx, valid_idx, n_epochs):
    log('... training', True)

    def get_n_batches(v):
        return len(v) / batch_size
    n_train_batches = get_n_batches(train_idx)
    n_valid_batches = get_n_batches(valid_idx)

    patience = 100000  # look as this many examples regardless
    patience_increase = 2
    improvement_threshold = 0.998
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    best_iter = 0

    try:
        for epoch in range(n_epochs):
            for minibatch_index in xrange(n_train_batches):
                train_model(train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])
                iteration = (epoch) * n_train_batches + minibatch_index

                if (iteration + 1) % validation_frequency == 0:
                    validation_losses = [validate_model(valid_idx[i * batch_size: (i + 1) * batch_size])
                                         for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    log(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.)
                    )

                    if this_validation_loss < best_validation_loss:
                        if (this_validation_loss <
                                best_validation_loss * improvement_threshold):
                            patience = max(patience, iteration * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iteration

                if patience <= iteration:
                    raise StopIteration('out of patience for training')
    except StopIteration:
        pass
    return best_validation_loss, best_iter, iteration


def run(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
        dataset_name='mnist.pkl.gz', batch_size=30, n_hidden=500, dropout_p=0.2):
    args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    arg_summary = ', '.join(['{0}={1}'.format(arg, eval(arg)) for arg in args])
    log(arg_summary)

    prepared_data = (
        prepare_data(dataset_name, batch_size=batch_size))

    f_train, f_validate, train_idx, valid_idx = (
        build_model(prepared_data, L1_reg=L1_reg, L2_reg=L2_reg,
                    n_hidden=n_hidden, dropout_p=dropout_p,
                    learning_rate=learning_rate))

    start_time = time.clock()
    best_validation_loss, best_iter, iteration = (
        train_model(f_train, f_validate, batch_size, train_idx, valid_idx,
                    n_epochs=n_epochs))
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
    parser.add_argument('--f', dest='file', type=str, default='data/task_data2.gz',
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
