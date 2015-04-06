"""Emotiv driver.
run accepts processed data as a text file, trains and validates the model.
convert accepts raw data from a directory of .mat files and pickles them into a Dataset object.

Usage:
    emotiv_driver.py run [-i <input>] [-m <model>] [-f <feature>...] [-c <cond>...] [options]
    emotiv_driver.py convert <directory> <output>

Options:
    -m <model>, --model=<model>
        The name of the model family to use [default: randomforest].
    -f <feature>, --feature=<feature>
        The names of features to use [default: raw].
    -c <cond>, --cond=<cond>
        The names of conditions to use [default: NegativeLowArousalPictures PositiveLowArousalPictures].
    -i <input>, --in=<input>
        The input data file to use. [default: data/emotiv_all.gz]
    -e <error>, --err=<error>
        The name for the log file to be generated.
    -o <output>, --out=<output>
        The name of the file to saved trained model parameters.
    -p <param_set>, --param_set=<param_set>
        The name of the parameter set to use [default: emotiv_wide_search4].
    -q, --quiet
        Suppress writing to log and output files.
    -t, --task_number=<ints>
        A counter representing the queue position of the current job [default: 0].
    -d <data_mode>, --data_mode=<data_mode>
        Way to arrange the data after loading it - normal or paired [default: normal].
"""

from __future__ import print_function, division

import os
import warnings
import cPickle as pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt

from learntools.libs.logger import gen_log_name, log_me, set_log_file
from learntools.emotiv.data import (prepare_data, convert_raw_data, load_raw_data, load_siegle_data,
                                    gen_featured_dataset, to_paired)
from learntools.emotiv.filter import filter_data
from learntools.emotiv.features import construct_feature_generator
from learntools.data import cv_split_binarized
import learntools.deploy.config as config

import release_lock
release_lock.release()  # TODO: use theano config instead. We have to figure out
# what they did with the config.compile.timeout variable because that's actually what we need


def smart_load_data(dataset_name=None, features=None, **kwargs):
    _, ext = os.path.splitext(dataset_name)
    if ext == '.mat':
        dataset = load_siegle_data(dataset_name=dataset_name, **kwargs)
    elif ext == '.gz' or ext == '.pickle':
        dataset = load_raw_data(dataset_name=dataset_name, **kwargs)
        filter_data(dataset)
        feature_generator = construct_feature_generator(features)
        dataset = gen_featured_dataset(dataset, feature_generator, **kwargs)
    elif ext == '.txt':
        dataset = prepare_data(dataset_name)
        filter_data(dataset, remove_suffix=True)
    else:
        raise ValueError
    return dataset


@log_me()
def run(task_num, model, output_name, data_mode, **kwargs):
    if model == 'multistage_batchnorm':
        from learntools.emotiv.multistage_batchnorm import run as multistage_batchnorm_run
        multistage_batchnorm_run(**kwargs)

    dataset = smart_load_data(**kwargs)
    train_idx, valid_idx = cv_split_binarized(dataset, percent=0.2, fold_index=task_num)
    if model == 'batchnorm':
        from learntools.emotiv.batchnorm import BatchNorm as SelectedModel
    elif model == 'conv':
        from learntools.emotiv.conv import ConvEmotiv as SelectedModel
    elif model == 'conv_batchnorm':
        from learntools.emotiv.batchnorm import ConvBatchNorm as SelectedModel
    elif model == 'svm':
        from learntools.emotiv.svm import SVM as SelectedModel
    elif model == 'randomforest':
        from learntools.emotiv.randomforest import RandomForest as SelectedModel
    elif model == 'knn':
        from learntools.emotiv.knn import KNN as SelectedModel
    elif model == 'ensemble':
        from learntools.emotiv.ensemble import LogRegEnsemble as SelectedModel
    else:
        raise ValueError("model type is not valid")

    prepared_data = (dataset, train_idx, valid_idx)
    if data_mode == 'normal':
        pass
    elif data_mode == 'paired':
        prepared_data = to_paired(prepared_data)
    else:
        raise ValueError("data_mode '{}' is not valid".format(data_mode))

    model = SelectedModel(prepared_data, **kwargs)
    model.train_full(**kwargs)
    with open('{}.model'.format(output_name), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    args = docopt(__doc__)

    # load args
    params = config.get_config(args['--param_set'])
    err_filename = args['--err'] or gen_log_name()
    out_filename = args['--out'] or "{}.model".format(err_filename)
    if args['--quiet']:
        log_filename = os.devnull
        out_filename = os.devnull
        print("Suppressing logging and output.")
    set_log_file(err_filename)

    task_num = int(args['--task_number'])

    params['model'] = args['--model']
    params['features'] = args['--feature']
    params['conds'] = args['--cond']
    params['dataset_name'] = args['--in']
    params['data_mode'] = args['--data_mode']
    params['output_name'] = out_filename
    params['task_num'] = int(args['--task_number'])

    if args['run']:
        run(**params)

    # TODO: figure out a good interface
    # elif model == 'multistage_pretrain':
    #     from learntools.emotiv.multistage_batchnorm import pretrain
    #     no_conds_params = combine_dict(params, {'conds': None})
    #     dataset = smart_load_data(**no_conds_params)
    #     train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
    #     full_data = (dataset, train_idx, valid_idx)
    #     pretrain(log_name=log_filename, full_data=full_data, **params)
    # elif model == 'multistage_tune':
    #     from learntools.emotiv.multistage_batchnorm import tune
    #     # find a param-file to load
    #     saved_weights = filter(lambda(fn): os.path.splitext(fn)[1] == '.weights', os.listdir('.'))
    #     selected_weight_file = saved_weights[random.randint(0, len(saved_weights) - 1)]
    #     dataset = smart_load_data(**params)
    #     train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
    #     prepared_data = (dataset, train_idx, valid_idx)
    #     tune(prepared_data=prepared_data, weight_file=selected_weight_file, **params)
    # elif model == 'multistage_randomforest':
    #     from learntools.emotiv.multistage_randomforest import run as mrf_run
    #     selected_weight_file = "2015_03_10_16_09_35_33122.log.weights"
    #     dataset = smart_load_data(**params)
    #     if 1:  # TODO: fix this
    #         train_idx, valid_idx = cv_split_randomized(dataset, percent=0.1, fold_index=task_num)
    #     else:
    #         train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    #     prepared_data = (dataset, train_idx, valid_idx)
    #     mrf_run(prepared_data=prepared_data, weight_file=selected_weight_file, **params)

    elif args['convert']:
        convert_raw_data(args['<directory>'], args['<output>'])

    else:
        raise Exception("Unknown command")
    
    print("Finished")