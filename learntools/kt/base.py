from learntools.libs.auc import auc
from learntools.model import Model


class BaseKT(Model):

    def evaluate(self, idxs, pred):
        '''scores the predictions of a given set of rows
        Args:
            idxs (int[]): the indices of the rows to be evaluated
            pred (float[]): the prediction for the label of that row
        Returns:
            float: an evaluation score (the higher the better)
        '''
        # _correct_y is int-casted, go to owner op (int-cast) to get shared variable
        # as first input and get its value without copying the value out
        _y = self._correct_y.owner.inputs[0].get_value(borrow=True)[idxs]
        return auc(_y[:len(pred)], pred, pos_label=1)

    def train(self, idxs, **kwargs):
        '''perform one iteration of training on some indices
        Args:
            idxs (int[]): the indices of the rows to be used in training
        Returns:
            (float, float[], int[]): a tuple of the loss, the predictions over the rows,
                and the row indices
        '''
        res = self._tf_train(idxs)
        return res[:3]

    def validate(self, idxs, **kwargs):
        '''perform one iteration of validation
        Args:
            idxs (int[]): the indices of the rows to be used in validation
        Returns:
            (float, float[], int[]): a tuple of the loss, the predictions over the rows,
                and the row indices
        '''
        res = self._tf_valid(idxs)
        return res[:3]
