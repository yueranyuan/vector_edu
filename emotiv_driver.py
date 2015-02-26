"""Emotiv driver.
run accepts processed data as a text file, trains and validates the model.
convert_raw accepts raw data from a directory of .mat files and pickles them
into a Dataset object stored in the output file.

Usage:
    emotiv_driver.py run [options]
    emotiv_driver.py run_raw [options]
    emotiv_driver.py convert_raw <directory> <output>
    emotiv_driver.py run_subject [options]
    emotiv_driver.py run_autoencoder [options]
    emotiv_driver.py run_batchnorm [options]
    emotiv_driver.py run_multistage [options]

Options:
    -p <param_set>, --param_set=<param_set>
        The name of the parameter set to use [default: emotiv_wide_search].
    -f <file>, --file=<file>
        The data file to use [default: raw_data/all_siegle.txt].
    -o <file>, --out=<file>
        The name for the log file to be generated.
    -q, --quiet
        Do not output to a log file.
    -t, --task_number=<task_num>
        A counter representing the queue position of the current job.
"""

from __future__ import print_function, division

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt

from learntools.libs.logger import gen_log_name, log_me, set_log_file
from learntools.emotiv.data import prepare_data, convert_raw_data, segment_raw_data, load_siegle_data
from learntools.data import cv_split
from learntools.data.crossvalidation import cv_split_within_column
import learntools.deploy.config as config

import release_lock
release_lock.release()  # TODO: use theano config instead. We have to figure out
# what they did with the config.compile.timeout variable because that's actually
# what we need


class ModelType(object):
    BASE = 0
    RAW_BASE = 1
    SUBJECT = 2
    AUTOENCODER = 3
    BATCH_NORM = 4

@log_me()
def run(task_num=0, model_type=ModelType.BASE, **kwargs):
    if model_type == ModelType.BASE:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
        dataset = prepare_data(**kwargs)
        train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    elif model_type == ModelType.RAW_BASE:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
        dataset = segment_raw_data(**kwargs)
        train_idx, valid_idx = cv_split(dataset, percent=0.50, fold_index=task_num)
    elif model_type == ModelType.SUBJECT:
        from learntools.emotiv.persubject import SubjectEmotiv as SelectedModel
        dataset = segment_raw_data(**kwargs)
        train_idx, valid_idx = cv_split_within_column(dataset, percent=0.25, fold_index=task_num, min_length=4,
                                                      key='subject')
    elif model_type == ModelType.AUTOENCODER:
        from learntools.emotiv.emotiv_autoencode import AutoencodeEmotiv as SelectedModel
        dataset = segment_raw_data(**kwargs)
        train_idx, valid_idx = cv_split(dataset, percent=0.50, fold_index=task_num)
    elif model_type == ModelType.BATCH_NORM:
        from learntools.emotiv.batchnorm import BatchNorm as SelectedModel
        dataset = load_siegle_data(**kwargs)
        train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    else:
        raise Exception("model type is not valid")
    prepared_data = (dataset, train_idx, valid_idx)

    model = SelectedModel(prepared_data, **kwargs)
    model.train_full(**kwargs)


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

    params['conds'] = ['EyesClosed', 'EyesOpen']

    if args['run']:
        run(task_num=0, model_type=ModelType.BASE, **params)
    elif args['run_raw']:
        run(task_num=0, model_type=ModelType.RAW_BASE, **params)
    elif args['convert_raw']:
        convert_raw_data(args['<directory>'], args['<output>'])
    elif args['run_subject']:
        run(task_num=0, model_type=ModelType.SUBJECT, **params)
    elif args['run_autoencoder']:
        run(task_num=0, model_type=ModelType.AUTOENCODER, **params)
    elif args['run_batchnorm']:
        run(task_num=0, model_type=ModelType.BATCH_NORM, **params)
    elif args['run_multistage']:
        from learntools.emotiv.multistage import run_multistage
        run_multistage(task_num=0, **params)
    
    print("Finished")
