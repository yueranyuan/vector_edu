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

from mlp import MLP
from vmlp import VectorLayer
import config
from utils import gen_log_name, make_shared, random_unique_subset


def log(txt, also_print=False):
    if also_print:
        print txt
    with open(LOG_FILE, 'a+') as f:
        f.write('{0}\n'.format(txt))


def prepare_data(dataset_name, batch_size):
    log('... loading data', True)

    with gzip.open(dataset_name, 'rb') as f:
        dataset = cPickle.load(f)
    skill_x, subject_x, correct_y = dataset

    validation = random_unique_subset(subject_x) & random_unique_subset(skill_x)
    train_idx, _ = numpy.nonzero(numpy.logical_not(validation))
    valid_idx, _ = numpy.nonzero(validation)

    skill_x = make_shared(skill_x)
    subject_x = make_shared(subject_x)
    correct_y = make_shared(correct_y, to_int=True)
    return (skill_x, subject_x, correct_y), (train_idx, valid_idx)


def build_model(prepared_data, L1_reg, L2_reg, n_hidden, dropout_p,
                learning_rate):
    log('... building the model', True)
    skill_x, subject_x, correct_y = prepared_data

    rng = numpy.random.RandomState(1234)
    skill_vector_len = 50
    indices = T.ivector('idx')
    y = correct_y[indices]
    skill_vectors = VectorLayer(rng=rng,
                                full_input=skill_x,
                                n_skills=4600,
                                indices=indices,
                                vector_length=skill_vector_len)
    subject_vector_len = 50
    subject_vectors = VectorLayer(rng=rng,
                                  full_input=subject_x,
                                  n_skills=4600,
                                  indices=indices,
                                  vector_length=subject_vector_len)

    classifier = MLP(rng=rng,
                     n_in=skill_vector_len + subject_vector_len,
                     input=T.concatenate([skill_vectors.output, subject_vectors.output], axis=1),
                     n_hidden=n_hidden,
                     n_out=3)

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    def gen_givens(data_x=None, data_y=None):
        return {
            y: data_y[indices],
            classifier.dropout: dropout_p
        }

    f_valid = theano.function(
        inputs=[indices],
        outputs=[classifier.errors(y)],
        givens={
            classifier.dropout: dropout_p
        },
        on_unused_input="ignore",
        allow_input_downcast=True
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    updates = updates + skill_vectors.get_updates(cost, learning_rate)
    f_train = theano.function(
        inputs=[indices],
        outputs=[cost],
        updates=updates,
        givens={
            classifier.dropout: dropout_p
        },
        on_unused_input="ignore",
        allow_input_downcast=True
    )
    return f_train, f_valid


def train_model(train_model, validate_model,
                batch_size, train_idx, valid_idx, n_epochs):
    log('... training', True)

    def get_n_batches(v):
        return len(v) / batch_size
    n_train_batches = get_n_batches(train_idx)
    n_valid_batches = get_n_batches(valid_idx)

    patience = 20000  # look as this many examples regardless
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


def run(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=10,
        dataset_name='mnist.pkl.gz', batch_size=30, n_hidden=500, dropout_p=0.2):
    args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    arg_summary = ', '.join(['{0}={1}'.format(arg, eval(arg)) for arg in args])
    log(arg_summary)

    prepared_data, (train_idx, valid_idx) = (
        prepare_data(dataset_name, batch_size=batch_size))

    f_train, f_validate = (
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
