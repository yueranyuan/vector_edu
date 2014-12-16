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
from utils import gen_log_name, make_shared


def log(txt, also_print=False):
    if also_print:
        print txt
    with open(LOG_FILE, 'a+') as f:
        f.write('{0}\n'.format(txt))


def prepare_data(dataset_name, batch_size):
    log('... loading data', True)

    with gzip.open(dataset_name, 'rb') as f:
        dataset = [make_shared(d) for d in cPickle.load(f)]
    skill_x, subject_x, correct_y = dataset
    correct_y = T.cast(correct_y, 'int32')

    n_train_batches = skill_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = skill_x.get_value(borrow=True).shape[0] / batch_size
    return (skill_x, subject_x, correct_y), (n_train_batches, n_valid_batches)


def build_model(prepared_data, batch_size, L1_reg, L2_reg, n_hidden, dropout_p,
                learning_rate):
    log('... building the model', True)
    skill_x, subject_x, correct_y = prepared_data

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
    skill_vector_len = 50
    indices = index + make_shared(range(batch_size), True)
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
        inputs=[index],
        outputs=[classifier.errors(y)],
        givens=gen_givens(data_y=correct_y),
        on_unused_input="ignore"
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    updates = updates + skill_vectors.get_updates(cost, learning_rate)
    f_train = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens=gen_givens(data_x=skill_x, data_y=correct_y),
        on_unused_input="ignore"
    )
    return f_train, f_valid


def train_model(train_model, validate_model, n_train_batches, n_valid_batches,
                n_epochs):
    log('... training', True)

    patience = 20000  # look as this many examples regardless
    patience_increase = 2
    improvement_threshold = 0.998
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    best_iter = 0

    try:
        for epoch in range(n_epochs):
            for minibatch_index in xrange(n_train_batches):
                train_model(minibatch_index)
                iter = (epoch) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
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
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    raise StopIteration('out of patience for training')
    except StopIteration:
        pass
    return best_validation_loss, best_iter


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset_name='mnist.pkl.gz', batch_size=30, n_hidden=500, dropout_p=0.2):
    args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    arg_summary = ', '.join(['{0}={1}'.format(arg, eval(arg)) for arg in args])
    log(arg_summary)

    prepared_data, (n_train_batches, n_valid_batches) = (
        prepare_data(dataset_name, batch_size=batch_size))

    f_train, f_validate = (
        build_model(prepared_data, batch_size=batch_size, L1_reg=L1_reg,
                    L2_reg=L2_reg, n_hidden=n_hidden, dropout_p=dropout_p,
                    learning_rate=learning_rate))

    start_time = time.clock()
    best_validation_loss, best_iter = (
        train_model(f_train, f_validate, n_train_batches, n_valid_batches,
                    n_epochs=n_epochs))
    end_time = time.clock()
    training_time = (end_time - start_time) / 60.

    log(('Optimization complete. Best validation score of %f %% '
         'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, 0.), True)
    log('Code ran for ran for %.2fm' % (training_time))
    return (best_validation_loss * 100., best_iter + 1, iter, training_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run a theano experiment on this computer")
    parser.add_argument('param_set', type=str,
                        choices=config.all_param_set_keys,
                        help='the name of the parameter set that we want to use')
    parser.add_argument('--f', dest='file', type=str, default='data/task_data2.gz',
                        help='the data file to use')
    parser.add_argument('-o', dest='outname', type=str, default=gen_log_name(),
                        help='name for the log file to be generated')
    args = parser.parse_args()

    params = config.get_config(args.param_set)
    LOG_FILE = args.outname
    log(test_mlp(dataset_name=args.file, **params))
    print "finished"
    if sys.platform.startswith('win'):
        from win_utils import winalert
        winalert()
