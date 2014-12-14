import os
import sys
import time
import inspect

import numpy

#from sklearn.metrics import roc_curve, auc
import theano
import theano.tensor as T
from vmlp import VMLP
from data import load_data
import datetime
from random import randint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import config
import argparse

LOG_FILE = '{time}_{nonce}.log'.format(
    time=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    nonce=str(randint(0, 99999)))


def log(txt):
    with open(LOG_FILE, 'a+') as f:
        f.write('{0}\n'.format(txt))


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=30, n_hidden=500, dropout_p=0.0):
    args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    arg_summary = ', '.join(['{0}={1}'.format(arg, eval(arg)) for arg in args])
    log(arg_summary)
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    log('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
    classifier = VMLP(
        rng=rng,
        input=x,
        n_skills=4600,
        vector_length=50,
        n_hidden=n_hidden,
        n_out=3,
        full_input=train_set_x
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), classifier.output],
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            classifier.dropout: dropout_p
        },
        mode='DebugMode'
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y)],
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            classifier.dropout: dropout_p
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    updates = updates + classifier.get_updates(cost, index, batch_size,
                                               learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            classifier.dropout: dropout_p
        },
        on_unused_input="ignore"
    )

    ###############
    # TRAIN MODEL #
    ###############
    log('... training')

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

    temp_auc = 0

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

                    # test it on the test set
                    # test_losses, preds = zip(*[test_model(i) for i
                    #                    in xrange(n_test_batches)])
                    # test_score = numpy.mean(test_losses)
                    # preds = numpy.array(preds).flatten()
                    # print len(preds), preds
                    # fpr, tpr, thresholds = roc_curve(
                    # test_set_y.owner.inputs[0].get_value(borrow=True)[:len(preds)],
                    #                                 preds, pos_label=2)
                    # temp_auc = auc(fpr, tpr)
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    log(('Optimization complete. Best validation score of %f %% '
         'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return (best_validation_loss * 100., best_iter + 1, test_score * 100., iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run a theano experiment on this computer")
    parser.add_argument('param_set', type=str,
                        choices=config.all_param_set_keys,
                        help='the name of the parameter set that we want to use')
    parser.add_argument('--f', dest='file', type=str, default='data/task_data.gz',
                        help='the data file to use')
    args = parser.parse_args()

    params = config.get_config(args.param_set)
    log(test_mlp(dataset=args.file, **params))
    print "finished"
    if sys.platform.startswith('win'):
        from win_utils import winalert
        winalert()
