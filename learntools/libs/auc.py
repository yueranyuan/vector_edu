# an auc library that doesn't require sklearn
# thanks to https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
# I made some changes to style to fit our code-base

import numpy as np


def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = np.array([0] * len(x))
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r


def auc(actual, posterior, pos_label=1):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    positives = np.equal(actual, pos_label)
    num_positive = sum(positives)
    num_negative = len(actual) - num_positive
    sum_positive = sum(r[positives])
    auc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) /
           (num_negative * num_positive))
    return auc

if __name__ == "__main__":
    from sklearn import metrics
    import random
    y = np.array([1] * random.randint(1, 20) + [2] * random.randint(1, 20))
    pred = np.array([random.random() for i in xrange(len(y))])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    standard = metrics.auc(fpr, tpr)
    custom = auc(y, pred, pos_label=2)
    if standard != custom:
        raise Exception("custom AUC doesn't match SKlearn AUC, {0} != {1}".format(
            standard, custom))
