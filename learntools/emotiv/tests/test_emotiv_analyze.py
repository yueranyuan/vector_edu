from __future__ import division
from random import shuffle, random, randint
import itertools

import numpy as np

from learntools.emotiv.analyze import analyze_eeg_features
from learntools.data import Dataset


def generate_random_data(n=5000, n_good_features=10, n_bad_features=10):
    conds = [randint(0, 1) for i in xrange(n)]

    def generate_good_feature():
        source_values = [random(), random()]
        return np.asarray([source_values[c] for c in conds])

    def generate_bad_feature():
        return np.ones(n) * random()

    good_features = [generate_good_feature() for i in xrange(n_good_features)]
    bad_features = [generate_bad_feature() for i in xrange(n_bad_features)]

    goodness_feature_pairs = ([(True, feature) for feature in good_features] +
                              [(False, feature) for feature in bad_features])
    shuffle(goodness_feature_pairs)

    goodness = [good for (good, feature) in goodness_feature_pairs]
    features = [feature for (good, feature) in goodness_feature_pairs]
    features = np.asarray(features).T

    features += np.random.random(features.shape) * 0.4  # add noise

    n_features = n_good_features + n_bad_features
    significant_idx = list(itertools.compress(xrange(n_features), goodness))
    return conds, features, significant_idx


def test_analyze_random():
    """ Run analysis a many times, it must be right at least 50% of the time
    """
    n_attempts = 100
    results = [True] * n_attempts
    for i in xrange(n_attempts):
        n_good_features = randint(3, 7)
        n_bad_features = randint(3, 7)
        data = Dataset((('condition', Dataset.ENUM), ('eeg', Dataset.MATFLOAT)),
                       n_rows=randint(2000, 4000))
        conds, features, significant_idx = generate_random_data(n=data.n_rows,
                                                                n_good_features=n_good_features,
                                                                n_bad_features=n_bad_features)
        for data_i, row in enumerate(itertools.izip(conds, features)):
            data[data_i] = row

        significant_features, _, remaining_features = analyze_eeg_features(data, plot=False, silent=True)
        try:
            assert significant_features == set(significant_idx)
            assert remaining_features == (set(xrange(n_good_features + n_bad_features))
                                          - significant_features)
        except AssertionError:
            print results, i
            results[i] = False

    assert sum(results) / n_attempts > 0.5