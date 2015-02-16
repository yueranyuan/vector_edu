"""Emotiv driver.
run accepts processed data as a text file, trains and validates the model.
convert_raw accepts raw data from a directory of .mat files and pickles them
into a Dataset object stored in the output file.

Usage:
    emotiv_driver.py run [--cond=<cond>]... [options]
    emotiv_driver.py run_raw [--cond=<cond>]... [options]
    emotiv_driver.py convert_raw <directory> <output>

Options:
    -p <param_set>, --param_set=<param_set>
        The name of the parameter set to use [default: emotiv_wide_search2].
    -f <file>, --file=<file>
        The data file to use [default: raw_data/all_siegle.txt].
    -o <file>, --out=<file>
        The name for the log file to be generated.
    -q, --quiet
        Do not output to a log file.
    -t, --task_number=<task_num>
        A counter representing the queue position of the current job.
    -c, --cond=<cond>
        String representing a condition to include [default: EyesOpen EyesClosed].
"""

from __future__ import print_function, division

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt

from learntools.libs.logger import gen_log_name, log_me, set_log_file
from learntools.emotiv.data import prepare_data, convert_raw_data, segment_raw_data
from learntools.data import cv_split, Dataset
import learntools.deploy.config as config

import release_lock
release_lock.release()  # TODO: use theano config instead. We have to figure out
# what they did with the config.compile.timeout variable because that's actually
# what we need

class ModelType(object):
    BASE = 0
    RAW_BASE = 1

@log_me()
def run(task_num=0, model_type=ModelType.BASE, **kwargs):
    if model_type == ModelType.BASE:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
        dataset = prepare_data(**kwargs)
    elif model_type == ModelType.RAW_BASE:
        from learntools.emotiv.base import BaseEmotiv as SelectedModel
        dataset = segment_raw_data(**kwargs)
    else:
        raise Exception("model type is not valid")
    train_idx, valid_idx = cv_split(dataset, percent=0.1, fold_index=task_num)
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

    params['conds'] = args['--cond']
    print("Conditions:", args['--cond'])

    if args['run']:
        run(task_num=0, model_type=ModelType.BASE, **params)
    elif args['run_raw']:
        run(task_num=0, model_type=ModelType.RAW_BASE, **params)
    elif args['convert_raw']:
        convert_raw_data(args['<directory>'], args['<output>'])
    
    print("Finished")
