from itertools import imap, chain, groupby, islice

from learntools.libs.logger import log_me


class Model(object):
    @log_me('... building the model')
    # TODO: don't log self
    def __init__(self, *args, **kwargs):
        raise Exception("build model not implemented for this model")

    def evaluate(self, idx, pred):
        raise Exception("evaluate not implemented for this model")

    def train_evaluate(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def valid_evaluate(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def validate(self, idx, **kwargs):
        raise Exception("validate not implemented for this model")

    def train(self, idx, **kwargs):
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

    @property
    def valid_batches(self):
        try:
            return self._valid_batches
        except AttributeError:
            raise Exception("valid_batches not defined for this model")

    @valid_batches.setter
    def valid_batches(self, valid_batches):
        self._valid_batches = valid_batches


def gen_batches(idxs, keys, batch_size=None):
    all_keys = zip(*keys)
    batches = [list(islice(idx, 2, None)) for k, idx in
               groupby(idxs, key=lambda i: all_keys[i])]

    if batch_size:
        def sub_batch(idxs):
            return [idxs[i * batch_size: (i + 1) * batch_size] for i in
                    xrange(int(len(idxs) / batch_size))]
        batches = list(chain.from_iterable(imap(sub_batch, batches)))
    return batches
