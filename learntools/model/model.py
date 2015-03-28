import random

from itertools import groupby


class Model(object):
    '''a trainable, applyable model

    Attributes:
        train_batches (int[][]): a list of training batches. Each batch is a list of row
            indices
        valid_batches (int[][]): a list of validation batches. Each batch is a list of row
            indices
    '''
    # TODO: don't log self
    def __init__(self, *args, **kwargs):
        raise Exception("build model not implemented for this model")

    def evaluate(self, idx, preds):
        '''scores the predictions of a given set of rows

        Args:
            idxs (int[]): the indices of the rows to be evaluated
            pred (float[]): the prediction for the label of that row
        Returns:
            float: an evaluation score (the higher the better)
        '''
        raise Exception("evaluate not implemented for this model")

    def train_evaluate(self, idxs, preds, *args, **kwargs):
        '''scores the predictions of a given set of rows under training

        (see evaluate())
        '''
        return self.evaluate(idxs, preds, *args, **kwargs)

    def valid_evaluate(self, idxs, preds, *args, **kwargs):
        '''scores the predictions of a given set of rows under validation

        (see evaluate())
        '''
        return self.evaluate(idxs, preds, *args, **kwargs)

    def validate(self, idx, **kwargs):
        '''perform one iteration of validation

        Args:
            idxs (int[]): the indices of the rows to be used in validation
        Returns:
            (float, float[], int[]): a tuple of the loss, the predictions over the rows,
                and the row indices
        '''
        raise Exception("validate not implemented for this model")

    def train(self, idx, **kwargs):
        '''perform one iteration of training on some indices

        Args:
            idxs (int[]): the indices of the rows to be used in training
        Returns:
            (float, float[], int[]): a tuple of the loss, the predictions over the rows,
                and the row indices
        '''
        raise Exception("train not implemented for this model")

    @property
    def train_batches(self):
        try:
            return self._train_batches
        except AttributeError:
            raise Exception("train_batches not defined for this model")

    @train_batches.setter
    def train_batches(self, train_batches):
        self._train_batches = train_batches

    def gen_train(self, shuffle=False, loss=False, **kwargs):
        batch_order = range(len(self.train_batches))
        if shuffle:
            random.shuffle(batch_order)
        for i in batch_order:
            losses, preds, idxs = self.train(self.train_batches[i], **kwargs)
            if loss:
                yield idxs, losses
            else:
                yield idxs, preds

    @property
    def valid_batches(self):
        try:
            return self._valid_batches
        except AttributeError:
            raise Exception("valid_batches not defined for this model")

    @valid_batches.setter
    def valid_batches(self, valid_batches):
        self._valid_batches = valid_batches

    def gen_valid(self, shuffle=False, loss=False, **kwargs):
        batch_order = range(len(self.valid_batches))
        if shuffle:
            random.shuffle(batch_order)
        for i in batch_order:
            losses, preds, idxs = self.validate(self.valid_batches[i], **kwargs)
            if loss:
                yield idxs, losses
            else:
                yield idxs, preds

    def train_full(self, strategy=None, **kwargs):
        import time
        from learntools.model import train_model
        from learntools.libs.logger import log

        # use default strategy if no training strategy is provided
        if strategy is None:
            if hasattr(self, 'train_strategy'):
                strategy = self.train_strategy
            else:
                strategy = train_model

        start_time = time.clock()
        best_validation_loss, best_epoch = strategy(self, **kwargs)
        end_time = time.clock()
        training_time = (end_time - start_time) / 60.

        log(('Optimization complete. Best validation score of %f %%') %
            (best_validation_loss * 100.), True)
        log('Code ran for ran for %.2fm' % (training_time))
        return best_validation_loss, best_epoch


def gen_batches_by_keys(idxs, keys):
    '''breaks a provided group of rows into batches based on some input keys

    Args:
        idxs (int[]): the indices to break into batches
        keys (list[]): each distinct row in the keys will be broken into a separate
            batch. NOTE: Keys must be pre-sorted.

    Returns:
        int[][]: idx divided into batches

    Examples:
        >>> gen_batches_by_keys([0, 1, 2, 3], [[1, 2, 1, 1], [2, 1, 1, 1]])
        [[0], [1], [2, 3]]
    '''
    all_keys = zip(*keys)
    batches = [list(idx) for k, idx in
               groupby(idxs, key=lambda i: all_keys[i])]
    return batches


def gen_batches_by_size(idxs, batch_size):
    '''breaks a provided group of rows into batches based on batch size

    Args:
        idxs (int[]): the indices to break into batches
        batch_size (int): size of each batch

    Returns:
        int[][]: idx divided into batches

    Examples:
        >>> gen_batches_by_size([0, 1, 2, 3, 4], 2)
        [[0, 1], [2, 3]]
    '''
    if len(idxs) < batch_size:
      return [idxs]

    return [idxs[i * batch_size: (i + 1) * batch_size] for i in
            xrange(int(len(idxs) / batch_size))]
