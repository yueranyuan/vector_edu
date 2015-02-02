from itertools import imap, chain
import random

from learntools.libs.logger import log_me, log
from learntools.libs.utils import transpose


@log_me('... training')
def train_model(model, n_epochs=500, patience=50,
                patience_increase=40, improvement_threshold=1,
                validation_frequency=5, learning_rate=0.02,
                rng_seed=1023, **kwargs):
    best_valid_accuracy = 0
    best_epoch = 0

    train_model = model.train
    valid_model = model.validate
    train_batches = model.train_batches
    valid_batches = model.valid_batches
    train_eval = model.train_evaluate
    valid_eval = model.valid_evaluate

    # batch shuffling should be deterministic
    prev_rng_state = random.getstate()
    random.seed(rng_seed)

    def run_batches(model, batches, f_eval, shuffle=True, **kwargs):
        batch_order = range(len(batches))
        if shuffle:
            random.shuffle(batch_order)
        # Aaron: this is not really a speed critical part of the code but we
        # can come back and redo AUC in theano if we want to make this suck less
        results = imap(lambda i: model(batches[i], **kwargs), batch_order)
        (losses, preds, idxs) = transpose(results)
        accuracy = f_eval(list(chain.from_iterable(idxs)),
                          list(chain.from_iterable(preds)))
        return accuracy, losses

    for epoch in range(n_epochs):
        train_accuracy, train_losses = run_batches(train_model, train_batches, train_eval,
                                                   learning_rate=learning_rate)
        log('epoch {epoch}, train accuracy {err:.2%}'.format(
            epoch=epoch, err=train_accuracy), True)

        if (epoch + 1) % validation_frequency == 0:
            valid_accuracy, _ = run_batches(valid_model, valid_batches, valid_eval, shuffle=False)
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
