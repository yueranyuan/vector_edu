"""Emotiv driver. Accepts raw data as a text file.

Usage:
  emotiv_driver.py [options]

Options:
  -p <param_set>, --param_set=<param_set>
    The name of the parameter set to use [default: default].
  -f <file>, --file=<file>
    The data file to use.
  -o <file>, --out=<file>
    The name for the log file to be generated.
  -q, --quiet
    Do not output to a log file.
"""

from __future__ import print_function, division

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from docopt import docopt
import numpy as np

from learntools.libs.utils import mask_to_idx
from learntools.libs.logger import gen_log_name, log_me, log, set_log_file
from learntools.emotiv.data import prepare_data
from learntools.model.train import train_model
import learntools.deploy.config as config


@log_me()
def run(model_type=0, **kwargs):
  if model_type == 0:
    from learntools.emotiv.base import BaseEmotiv as SelectedModel
  else:
    raise Exception("model type is not valid")

  dataset = prepare_data(**kwargs)

  # generate the train-validation split
  validation_ratio = 0.1
  rng = np.random.RandomState(1337)
  rvec = rng.random_sample(len(dataset))
  train_idx = mask_to_idx(rvec >= validation_ratio)
  valid_idx = mask_to_idx(rvec < validation_ratio)
  prepared_data = (dataset, train_idx, valid_idx)

  model = SelectedModel(prepared_data, **kwargs)
  model.train_full()


if __name__ == '__main__':
  default_dataset = 'raw_data/all_siegle.txt'

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

  if sys.platform.startswith('win'):
    from win_utils import winalert
    winalert()
