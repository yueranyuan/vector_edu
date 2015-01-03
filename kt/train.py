import inspect
from itertools import imap, chain
import random

import numpy as np

from libs.logger import log, log_args
from libs.utils import transpose


def train_model(train_model, validate_model, train_batches, valid_batches,
                train_eval, valid_eval, n_epochs=300, patience=50,
                patience_increase=40, improvement_threshold=1,
                validation_frequency=5, learning_rate=0.02, **kwargs):
    log('... training', True)
    log_args(inspect.currentframe())

    best_valid_error = 0
    best_epoch = 0

    def run_batches(model, batches, f_eval, **kwargs):
        batch_order = range(len(batches))
        random.shuffle(batch_order)
        # Aaron: this is not really a speed critical part of the code but we
        # can come back and redo AUC in theano if we want to make this suck less
        results = imap(lambda i: model(batches[i], **kwargs), batch_order)
        (losses, preds, idxs) = transpose(results)
        error = f_eval(list(chain.from_iterable(idxs)),
                       list(chain.from_iterable(preds)))
        return error, losses

    for epoch in range(n_epochs):
        train_error, train_losses = run_batches(train_model, train_batches, train_eval,
                                                learning_rate=learning_rate)
        log('epoch {epoch}, train error {err:.2%}'.format(
            epoch=epoch, err=train_error), True)

        if (epoch + 1) % validation_frequency == 0:
            valid_error, _ = run_batches(validate_model, valid_batches, valid_eval)
            log('epoch {epoch}, validation error {err:.2%}'.format(
                epoch=epoch, err=valid_error), True)

            if valid_error > best_valid_error:
                if (valid_error > best_valid_error * improvement_threshold):
                    patience = max(patience, epoch + patience_increase)
                best_valid_error = valid_error
                best_epoch = epoch

            if patience <= epoch:
                break
    return best_valid_error, best_epoch
