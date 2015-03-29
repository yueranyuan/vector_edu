import cPickle as pickle

from learntools.libs.logger import log_me, log
from learntools.emotiv.multistage_batchnorm import AutoencodingBatchNorm
from learntools.data.dataset import Dataset
from learntools.emotiv.randomforest import RandomForest


def run(prepared_data=None, weight_file=None, **kwargs):
    log('loading weight file {}'.format(weight_file))
    with open(weight_file, 'rb') as f:
        weights = pickle.load(f)
    data, _, _ = prepared_data
    randomforest1 = RandomForest(prepared_data=prepared_data)
    randomforest1.train_full()

    autoencoder = AutoencodingBatchNorm(prepared_data=prepared_data, serialized=weights, classifier_depth=None)
    mapped = autoencoder.encode(range(data.n_rows))
    data.set_column('eeg', Dataset.MATFLOAT)
    for i, row in enumerate(mapped):
        data.get_column('eeg')[i] = row

    randomforest2 = RandomForest(prepared_data=prepared_data)
    randomforest2.train_full()