"""Emotiv driver. Accepts raw data as a text file.

Usage:
    emotiv_driver.py [options]

Options:
    -p <param_set>, --param_set=<param_set>
        The name of the parameter set to use [default: emotiv_wide_search].
    -f <file>, --file=<file>
        The data file to use.
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
from learntools.emotiv.data import prepare_data
from learntools.data import cv_split
import learntools.deploy.config as config

import release_lock
release_lock.release()  # TODO: use theano config instead. We have to figure out
# what they did with the config.compile.timeout variable because that's actually
# what we need


@log_me()
def run(task_num=0, model_type=0, **kwargs):
    if model_type == 0:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
    else:
        raise Exception("model type is not valid")

    dataset = prepare_data(**kwargs)
    train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
    prepared_data = (dataset, train_idx, valid_idx)

    model = SelectedModel(prepared_data, **kwargs)
    model.train_full()


if __name__ == '__main__':
    default_dataset = "raw_data/all_siegle.txt"

    args = docopt(__doc__)

    params = config.get_config(args['--param_set'])
    log_filename = args['--out'] or gen_log_name()
    if args['--quiet']:
        log_filename = os.devnull
        print("Not printing to log file.")
    set_log_file(log_filename)

    if args['--file']:
        params['dataset_name'] = args['--file']
    elif 'dataset_name' not in params:
        params['dataset_name'] = default_dataset

    params['conds'] = ['EyesClosed', 'EyesOpen']
    run(0, **params)
    print("Finished")
