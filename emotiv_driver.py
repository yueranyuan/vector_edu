"""Emotiv driver.
run accepts processed data as a text file, trains and validates the model.
convert_raw accepts raw data from a directory of .mat files and pickles them
into a Dataset object stored in the output file.

Usage:
    emotiv_driver.py run [options]
    emotiv_driver.py convert_raw <directory> <output>

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

from datetime import datetime
import os
import traceback
import glob
import cPickle as pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt

from learntools.libs.logger import gen_log_name, log_me, set_log_file
from learntools.emotiv.data import prepare_data
from learntools.data import cv_split, Dataset
from learntools.data.dataset import LISTEN_TIME_FORMAT
import learntools.deploy.config as config
from learntools.libs.utils import loadmat

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
    model.train_full(**kwargs)


def convert_raw(directory, output):
    raw_files = glob.glob(os.path.join(directory, '*.mat'))

    subjects = {}
    headers = [
        ('subject', Dataset.STR),
        ('eeg_sequence', Dataset.SEQFLOAT),
        ('condition', Dataset.SEQINT), # TODO is this the column type we really want?
        ('time', Dataset.TIME),
    ]

    n_rows = len(raw_files)
    ds = Dataset(headers, n_rows)

    for i, raw_filename in enumerate(raw_files):
        print(raw_filename)
        try:
            raw_file = loadmat(raw_filename)
            filename, extension = os.path.splitext(os.path.basename(raw_filename))
            p = raw_file['p']
            eeg_sequence = p['EEG']
            condition = p['OtherData'][2, :]
            dt = datetime(*tuple(p['hdr']['orig']['T0']))
            timestr = dt.strftime(LISTEN_TIME_FORMAT)

            ds[i] = (filename, eeg_sequence, condition, timestr)
        except Exception as e:
            traceback.print_exc()
            return # fail hard

    print(len(raw_files), "files loaded")
    
    with open(output, 'wb') as f:
        pickle.dump(ds.to_pickle(), f, protocol=pickle.HIGHEST_PROTOCOL)


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

    if args['run']:
        run(0, **params)
    elif args['convert_raw']:
        convert_raw(args['<directory>'], args['<output>'])
    
    print("Finished")
