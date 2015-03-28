"""Emotiv driver.
run accepts processed data as a text file, trains and validates the model.
convert_raw accepts raw data from a directory of .mat files and pickles them
into a Dataset object stored in the output file.

Usage:
    emotiv_driver.py [options]
    emotiv_driver.py run [options]
    emotiv_driver.py run_raw [options]
    emotiv_driver.py convert_raw <directory> <output>
    emotiv_driver.py run_subject [options]
    emotiv_driver.py run_autoencoder [options]
    emotiv_driver.py run_batchnorm [options]
    emotiv_driver.py run_conv [options]
    emotiv_driver.py run_convbatchnorm [options]
    emotiv_driver.py run_multistage [options]

Options:
    -p <param_set>, --param_set=<param_set>
        The name of the parameter set to use [default: emotiv_wide_search4].
    -f <file>, --file=<file>
        The data file to use [default: data/emotiv_all.gz].
    -o <file>, --out=<file>
        The name for the log file to be generated.
    -q, --quiet
        Do not output to a log file.
    -t, --task_number=<ints>
        A counter representing the queue position of the current job [default: 0].
    -m <model>, --model=<model>
        The name of the model family that we are using [default: randomforest].
"""

from __future__ import print_function, division

import os
import warnings
import random
import cPickle as pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt

from learntools.libs.utils import combine_dict
from learntools.libs.logger import gen_log_name, log_me, set_log_file, get_log_file
from learntools.emotiv.data import (prepare_data, convert_raw_data, segment_raw_data, load_siegle_data,
                                    gen_wavelet_features, gen_fft_features)
from learntools.emotiv.filter import filter_data
from learntools.data import cv_split, cv_split_randomized
from learntools.data.crossvalidation import cv_split_within_column
import learntools.deploy.config as config

import release_lock
release_lock.release()  # TODO: use theano config instead. We have to figure out
# what they did with the config.compile.timeout variable because that's actually
# what we need

'''
COND_TYPES = [
        ['EyesClosed', 'EyesOpen'],
        ["PositiveLowArousalPictures", "NegativeLowArousalPictures"],
        ["PositiveHighArousalPictures", "NegativeHighArousalPictures"],
        ["PositiveHighArousalPictures", "PositiveLowArousalPictures"],
        ["NegativeHighArousalPictures", "NegativeLowArousalPictures"]]
'''

COND_TYPES = [
        ["PositiveLowArousalPictures", "NegativeLowArousalPictures"],
        ["PositiveHighArousalPictures", "PositiveLowArousalPictures"]]


def smart_load_data(dataset_name=None, feature_type='wavelet', duration=10, wavelet_depth=5, wavelet_family=3, data_name=0, **kwargs):
    #if data_name == 0:
    #    dataset_name = 'raw_data/sg_matlab.mat'
    #else:
    #    dataset_name = dataset_name
    _, ext = os.path.splitext(dataset_name)
    if ext == '.mat':
        dataset = load_siegle_data(dataset_name=dataset_name, **kwargs)
    elif ext == '.txt':
        dataset = prepare_data(dataset_name=dataset_name, **kwargs)
        filter_data(dataset, remove_suffix=True)
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


class ModelType(object):
    BASE = 0
    RAW_BASE = 1
    SUBJECT = 2
    AUTOENCODER = 3
    BATCH_NORM = 4
    SVM = 5
    MULTISTAGE_BATCH_NORM = 6
    RANDOMFOREST = 7
    ENSEMBLE = 8
    CONV_BATCH_NORM = 9
    CONV = 10


@log_me()
def run(task_num=0, cv_rand=0, model_type=ModelType.BASE, **kwargs):
    # task_num = 0  # KEEP AT 0 FOR ENSEMBLE
    if model_type == ModelType.BASE:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
        dataset = prepare_data(**kwargs)
        train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
    elif model_type == ModelType.RAW_BASE:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split_randomized(dataset, percent=0.50, fold_index=task_num)
    elif model_type == ModelType.SUBJECT:
        from learntools.emotiv.persubject import SubjectEmotiv as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split_within_column(dataset, percent=0.25, fold_index=task_num, min_length=4,
                                                      key='subject')
    elif model_type == ModelType.AUTOENCODER:
        from learntools.emotiv.emotiv_autoencode import AutoencodeEmotiv as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split_randomized(dataset, percent=0.10, fold_index=task_num)
    elif model_type == ModelType.BATCH_NORM:
        from learntools.emotiv.batchnorm import BatchNorm as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    elif model_type == ModelType.CONV_BATCH_NORM:
        from learntools.emotiv.batchnorm import ConvBatchNorm as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    elif model_type == ModelType.CONV:
        from learntools.emotiv.conv import ConvEmotiv as SelectedModel
        dataset = smart_load_data(**kwargs)
        if cv_rand:
            train_idx, valid_idx = cv_split_randomized(dataset, percent=0.2, fold_index=task_num)
        else:
            train_idx, valid_idx = cv_split(dataset, percent=0.2, fold_index=task_num)
    elif model_type == ModelType.SVM:
        from learntools.emotiv.svm import SVM as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split_randomized(dataset, percent=0.2, fold_index=task_num)
    elif model_type == ModelType.RANDOMFOREST:
        from learntools.emotiv.randomforest import RandomForest as SelectedModel
        dataset = smart_load_data(**kwargs)
        if cv_rand:
            train_idx, valid_idx = cv_split_randomized(dataset, percent=0.2, fold_index=task_num)
        else:
            train_idx, valid_idx = cv_split(dataset, percent=0.2, fold_index=task_num)
    elif model_type == ModelType.ENSEMBLE:
        from learntools.emotiv.ensemble import LogRegEnsemble as SelectedModel
        dataset = smart_load_data(**kwargs)
        train_idx, valid_idx = cv_split_randomized(dataset, percent=0.2, fold_index=task_num)
    else:
        raise Exception("model type is not valid")
    prepared_data = (dataset, train_idx, valid_idx)

    model = SelectedModel(prepared_data, **kwargs)
    _, params = model.train_full(**kwargs)
    pickle.dump({'model_type': model_type, 'params': params}, open("{log_name}.params".format(log_name=get_log_file()), "wb"))

'''
def build_batch_norm(task_num, **kwargs):
    import numpy as np
    from learntools.emotiv.batchnorm import BatchNormClassifier
    dataset = load_siegle_data(**kwargs)
    train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    xs = dataset.get_data('eeg')
    ys = dataset.get_data('condition')
    classifier = BatchNormClassifier(n_in=xs.shape[1], n_out=len(np.unique(ys)), **kwargs)
    classifier.fit(xs, ys, train_idx, valid_idx)
'''

if __name__ == '__main__':
    args = docopt(__doc__)

    params = config.get_config(args['--param_set'])
    log_filename = args['--out'] or gen_log_name()
    if args['--quiet']:
        log_filename = os.devnull
        print("Not printing to log file.")
    set_log_file(log_filename)

    if args['--file']:
        params['dataset_name'] = args['--file']

    task_num = int(args['--task_number'])

    params['conds'] = COND_TYPES[task_num % len(COND_TYPES)]

    if args['run']:
        run(task_num=task_num, model_type=ModelType.BASE, **params)
    elif args['run_raw']:
        run(task_num=task_num, model_type=ModelType.RAW_BASE, **params)
    elif args['convert_raw']:
        convert_raw_data(args['<directory>'], args['<output>'])
    elif args['run_subject']:
        run(task_num=task_num, model_type=ModelType.SUBJECT, **params)
    elif args['run_autoencoder']:
        run(task_num=task_num, model_type=ModelType.AUTOENCODER, **params)
    elif args['run_batchnorm']:
        run(task_num=task_num, model_type=ModelType.BATCH_NORM, **params)
    elif args['run_conv']:
        run(task_num=task_num, model_type=ModelType.CONV, **params)
    elif args['run_convbatchnorm']:
        run(task_num=task_num, model_type=ModelType.CONV_BATCH_NORM, **params)
    elif args['run_multistage']:
        from learntools.emotiv.multistage import run_multistage
        run_multistage(task_num=task_num, **params)
    else:
        # allowing us to select models with a flag without deprecating the original format
        model = args['--model']
        if model == 'batchnorm':
            run(task_num=task_num, model_type=ModelType.BATCH_NORM, **params)
        elif model == 'svm':
            run(task_num=task_num, model_type=ModelType.SVM, **params)
        elif model == 'randomforest':
            run(task_num=task_num, model_type=ModelType.RANDOMFOREST, **params)
        elif model == 'ensemble':
            run(task_num=task_num, model_type=ModelType.ENSEMBLE, **params)
        elif model == 'multistage_batchnorm':
            from learntools.emotiv.multistage_batchnorm import run as multistage_batchnorm_run
            no_conds_params = combine_dict(params, {'conds': None})
            dataset = smart_load_data(**no_conds_params)
            train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
            prepared_data = (dataset, train_idx, valid_idx)
            multistage_batchnorm_run(prepared_data=prepared_data, **params)
        elif model == 'multistage_pretrain':
            from learntools.emotiv.multistage_batchnorm import pretrain
            no_conds_params = combine_dict(params, {'conds': None})
            dataset = smart_load_data(**no_conds_params)
            train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
            full_data = (dataset, train_idx, valid_idx)
            pretrain(log_name=log_filename, full_data=full_data, **params)
        elif model == 'multistage_tune':
            from learntools.emotiv.multistage_batchnorm import tune
            # find a param-file to load
            saved_weights = filter(lambda(fn): os.path.splitext(fn)[1] == '.weights', os.listdir('.'))
            selected_weight_file = saved_weights[random.randint(0, len(saved_weights) - 1)]
            selected_weight_file = "2015_03_09_22_48_46_29694.log.weights"
            dataset = smart_load_data(**params)
            train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
            prepared_data = (dataset, train_idx, valid_idx)
            tune(prepared_data=prepared_data, weight_file=selected_weight_file, **params)
        elif model == 'multistage_randomforest':
            from learntools.emotiv.multistage_randomforest import run as mrf_run
            selected_weight_file = "2015_03_10_16_09_35_33122.log.weights"
            dataset = smart_load_data(**params)
            if 1:  # TODO: fix this
                train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
            else:
                train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
            prepared_data = (dataset, train_idx, valid_idx)
            mrf_run(prepared_data=prepared_data, weight_file=selected_weight_file, **params)
        else:
            raise Exception("invalid model family")
    
    print("Finished")
