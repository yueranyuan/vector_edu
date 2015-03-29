"""Emotiv driver.
run accepts processed data as a text file, trains and validates the model.
convert_raw accepts raw data from a directory of .mat files and pickles them
into a Dataset object stored in the output file.

Usage:
    ensemble_driver.py [options]

Options:
    -f <file>, --file=<file>
        The data file to use [default: raw_data/kkchang_matlab_fixed.mat].
    -o <file>, --out=<file>
        The name for the log file to be generated.
    -q, --quiet
        Do not output to a log file.
    -t, --task_number=<ints>
        A counter representing the queue position of the current job [default: 0].
"""

from __future__ import print_function, division

import os
import warnings
import cPickle as pickle
import random
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt

from learntools.libs.logger import gen_log_name, log_me, set_log_file, get_log_file
from learntools.emotiv.data import (prepare_data, segment_raw_data, load_siegle_data,
                                    gen_wavelet_features, gen_fft_features)
from learntools.emotiv.filter import filter_data
from learntools.data import cv_split, cv_split_randomized
import learntools.deploy.config as config

import release_lock
release_lock.release()  # TODO: use theano config instead. We have to figure out
# what they did with the config.compile.timeout variable because that's actually
# what we need


COND_TYPES = [
        ["PositiveLowArousalPictures", "NegativeLowArousalPictures"],
        ["PositiveHighArousalPictures", "PositiveLowArousalPictures"]]


def smart_load_data(dataset_name=None, feature_type='wavelet', duration=10, wavelet_depth=5, wavelet_family=3, **kwargs):
    _, ext = os.path.splitext(dataset_name)
    if ext == '.mat':
        dataset = load_siegle_data(dataset_name=dataset_name, **kwargs)
    elif ext == '.txt':
        dataset = prepare_data(dataset_name=dataset_name, **kwargs)
        filter_data(dataset)
    elif ext == '.gz' or ext == '.pickle':
        dataset = segment_raw_data(dataset_name=dataset_name, **kwargs)
        if feature_type == 'wavelet':
            dataset = gen_wavelet_features(dataset, duration=duration, sample_rate=128, depth=wavelet_depth, min_length=3,
                                           max_length=4, family='db{}'.format(wavelet_family), **kwargs)
        elif feature_type == 'fft':
            dataset = gen_fft_features(dataset, duration=duration, sample_rate=128, **kwargs)
        else:
            raise Exception('invalid feature type: {}'.format(feature_type))
        filter_data(dataset)
    else:
        raise ValueError
    return dataset


@log_me()
def run(task_num=0, cv_rand=0, **kwargs):
    from learntools.emotiv.ensemble import Ensemble as SelectedModel

    # prepare data
    dataset = smart_load_data(**kwargs)
    if cv_rand:
        train_idx, valid_idx = cv_split_randomized(dataset, percent=0.2, fold_index=task_num)
    else:
        train_idx, valid_idx = cv_split(dataset, percent=0.2, fold_index=task_num)
    prepared_data = (dataset, train_idx, valid_idx)

    # load classifiers to build ensemble out of
    saved_classifiers = filter(lambda(fn): os.path.splitext(fn)[1] == '.params', os.listdir('.'))

    # build and train ensemble
    model = SelectedModel(prepared_data, saved_classifiers=saved_classifiers, **kwargs)
    _, params = model.train_full(**kwargs)

if __name__ == '__main__':
    args = docopt(__doc__)

    params = {}
    log_filename = args['--out'] or gen_log_name()
    if args['--quiet']:
        log_filename = os.devnull
        print("Not printing to log file.")
    set_log_file(log_filename)

    if args['--file']:
        params['dataset_name'] = args['--file']

    task_num = int(args['--task_number'])

    params['conds'] = COND_TYPES[task_num % len(COND_TYPES)]
    run(task_num=task_num, **params)
    
    print("Finished")
