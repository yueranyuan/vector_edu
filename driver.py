import os
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
from utils import gen_log_name


def log(txt, also_print=False):
    if also_print:
        print txt
    with open(LOG_FILE, 'a+') as f:
        f.write('{0}\n'.format(txt))


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset_name='mnist.pkl.gz', batch_size=30, n_hidden=500, dropout_p=0.2):
    args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    arg_summary = ', '.join(['{0}={1}'.format(arg, eval(arg)) for arg in args])
    log(arg_summary)

    ##############
    # LOAD DATA  #
    ##############
    print '... loading data'

    def make_shared(d, to_int=False):
        sd = theano.shared(numpy.asarray(d, dtype=theano.config.floatX), borrow=True)
        if to_int:
            return T.cast(sd, 'int32')
        return sd
    with gzip.open(dataset_name, 'rb') as f:
        dataset = [make_shared(d) for d in cPickle.load(f)]
    skill_x, subject_x, correct_y = dataset
    correct_y = T.cast(correct_y, 'int32')

    train_set_x, train_set_y = skill_x, correct_y
    valid_set_x, valid_set_y = skill_x, correct_y
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    ###############
    # BUILD MODEL #
    ###############
    log('... building the model', True)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
    skill_vector_len = 50
    skill_vectors = VectorLayer(rng=rng,
                                full_input=skill_x,
                                n_skills=4600,
                                index=index,
                                batch_size=batch_size,
                                vector_length=skill_vector_len)
    subject_vector_len = 50
    subject_vectors = VectorLayer(rng=rng,
                                  full_input=subject_x,
                                  n_skills=4600,
                                  index=index,
                                  batch_size=batch_size,
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

    def gen_givens(data_x, data_y):
        return {
            y: data_y[index * batch_size:(index + 1) * batch_size],
            classifier.dropout: dropout_p
        }

    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y)],
        givens=gen_givens(valid_set_x, valid_set_y),
        on_unused_input="ignore"
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    updates = updates + skill_vectors.get_updates(cost, index, batch_size, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens=gen_givens(train_set_x, train_set_y),
        on_unused_input="ignore"
    )

    ###############
    # TRAIN MODEL #
    ###############
    log('... training', True)

    patience = 20000  # look as this many examples regardless
    patience_increase = 2
    improvement_threshold = 0.998
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

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
                done_looping = True
                break

    end_time = time.clock()
    log(('Optimization complete. Best validation score of %f %% '
         'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.), True)
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return (best_validation_loss * 100., best_iter + 1, test_score * 100., iter)

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
