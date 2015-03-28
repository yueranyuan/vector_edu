from __future__ import division
from itertools import izip
import random

from learntools.libs.logger import log_me, log
from learntools.libs.utils import transpose, flatten

ACCURACY_WINDOW = 7

@log_me('... training')
def train_model(model, n_epochs=500, patience=50,
                patience_increase=40, improvement_threshold=1,
                validation_frequency=5, learning_rate=0.02,
                rng_seed=1023, train_with_loss=False, **kwargs):
    best_valid_accuracy = 0
    best_epoch = 0

    # batch shuffling should be deterministic
    prev_rng_state = random.getstate()
    random.seed(rng_seed)

    def accuracy_from_results(results, eval_func):
        idxs, outs = transpose(results)
        if train_with_loss:
            accuracy = sum([len(_idxs) * loss for _idxs, loss in izip(idxs, outs)])
        else:
            idxs, preds = flatten(idxs), flatten(outs)
            accuracy = eval_func(idxs, preds)
        return accuracy

    valid_accuracy_window = []
    for epoch in range(n_epochs):
        results = list(model.gen_train(shuffle=True, loss=train_with_loss, learning_rate=learning_rate))
        train_accuracy = accuracy_from_results(results, eval_func=model.train_evaluate)
        log('epoch {epoch}, train accuracy {err:.2%}'.format(
            epoch=epoch, err=train_accuracy), True)

        if (epoch + 1) % validation_frequency == 0:
            results = list(model.gen_valid(shuffle=False, loss=train_with_loss))
            valid_accuracy = accuracy_from_results(results, eval_func=model.valid_evaluate)
            log('epoch {epoch}, validation accuracy {acc:.2%}'.format(
                epoch=epoch, acc=valid_accuracy), True)
            valid_accuracy_window.append(valid_accuracy)
            if len(valid_accuracy_window) > ACCURACY_WINDOW:
                valid_accuracy_window = valid_accuracy_window[-ACCURACY_WINDOW:]
            rolling_valid_accuracy = sum(valid_accuracy_window) / len(valid_accuracy_window)

            if rolling_valid_accuracy > best_valid_accuracy:
                if rolling_valid_accuracy > best_valid_accuracy * improvement_threshold:
                    patience = max(patience, epoch + patience_increase)
                best_valid_accuracy = rolling_valid_accuracy
                best_epoch = epoch

            if patience <= epoch:
                break

    # restore rng state
    random.setstate(prev_rng_state)

    return best_valid_accuracy, best_epoch