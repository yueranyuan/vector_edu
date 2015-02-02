import theano
import theano.tensor as T

from itertools import chain

from learntools.libs.logger import log_me
from learntools.libs.auc import auc
from learntools.model.logistic import LogisticRegression
from learntools.model.theano_utils import make_shared
from learntools.model import Model, gen_batches_by_size


class SimpleEmotiv(Model):
    @log_me('...building SimpleEmotiv')
    def __init__(self, prepared_data, batch_size=30, learning_rate=0.02, **kwargs):
        """
        Args:
            prepared_data : (Dataset, [int], [int])
                a tuple that holds the data to be used, the row indices of the
                training set, and the row indices of the validation set
            batch_size : int
                The size of the batches used to train
        """
        # 1: Organize data into batches
        ds, train_idx, valid_idx = prepared_data
        input_size = ds.get_data('eeg').shape[1]

        self._xs = make_shared(ds.get_data('eeg'), name='eeg')
        self._ys = make_shared(ds.get_data('condition'), to_int=True, name='condition')

        self.train_batches = gen_batches_by_size(train_idx, batch_size)
        self.valid_batches = gen_batches_by_size(valid_idx, 1)

        # 2: Connect the model
        classifier = LogisticRegression(n_in=input_size,
                                        n_out=2)

        input_idxs = T.ivector('input_idxs')
        classifier_input = self._xs[input_idxs]
        classifier_input.name = 'classifier_input'
        pY = classifier.instance(classifier_input)
        true_y = self._ys[input_idxs]
        true_y.name = 'true_y'

        # 3: Create theano functions
        loss = -T.mean(T.log(pY)[T.arange(input_idxs.shape[0]), true_y])
        loss.name = 'loss'
        subnets = [classifier]
        cost = loss
        cost.name = 'overall_cost'

        func_args = {
            'inputs': [input_idxs],
            'outputs': [loss, pY[:, 1] - pY[:, 0], input_idxs],
            'allow_input_downcast': True,
        }
        params = chain.from_iterable(net.params for net in subnets)
        update_parameters = [(param, param - learning_rate * T.grad(cost, param))
                             for param in params]

        self._tf_valid = theano.function(**func_args)
        self._tf_train = theano.function(
            updates=update_parameters,
            **func_args)

    def evaluate(self, idxs, pred):
        y = self._ys.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(y[:len(pred)], pred, pos_label=1)

    def validate(self, idxs, **kwargs):
        return self._tf_valid(idxs)

    def train(self, idxs, **kwargs):
        return self._tf_train(idxs)
