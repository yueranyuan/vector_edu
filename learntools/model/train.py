from __future__ import division
from itertools import imap, chain
import random

from learntools.libs.logger import log_me, log
from learntools.libs.utils import transpose, flatten


@log_me('... training')
def train_model(model, n_epochs=500, patience=50,
                patience_increase=40, improvement_threshold=1,
                validation_frequency=5, learning_rate=0.02,
                rng_seed=1023, **kwargs):
    best_valid_accuracy = 0
    best_epoch = 0

    # batch shuffling should be deterministic
    prev_rng_state = random.getstate()
    random.seed(rng_seed)

    def run_epoch(gen_batch, shuffle=True, **kwargs):
        return list(gen_batch())
        sum_acc = 0
        sum_n = 0
        for n, acc in gen_batch(shuffle=shuffle, **kwargs):
            sum_acc += acc * n
            sum_n += n
        return sum_acc / sum_n

    def aggregate_epoch_results(results):
        idxs, preds = transpose(results)
        return flatten(idxs), flatten(preds)

    for epoch in range(n_epochs):
        results = list(model.gen_train(shuffle=True, learning_rate=learning_rate))
        idxs, preds = aggregate_epoch_results(results)
        train_accuracy = model.train_evaluate(idxs, preds)
        log('epoch {epoch}, train accuracy {err:.2%}'.format(
            epoch=epoch, err=train_accuracy), True)

        if (epoch + 1) % validation_frequency == 0:
            results = list(model.gen_valid(shuffle=False))
            idxs, preds = aggregate_epoch_results(results)
            valid_accuracy = model.train_evaluate(idxs, preds)
            log('epoch {epoch}, validation accuracy {acc:.2%}'.format(
                epoch=epoch, acc=valid_accuracy), True)

            if valid_accuracy > best_valid_accuracy:
                if (valid_accuracy > best_valid_accuracy * improvement_threshold):
                    patience = max(patience, epoch + patience_increase)
                best_valid_accuracy = valid_accuracy
                best_epoch = epoch

            if patience <= epoch:
                break

    # restore rng state
    random.setstate(prev_rng_state)

    return best_valid_accuracy, best_epoch
