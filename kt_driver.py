import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from learntools.libs.logger import gen_log_name, log_me, set_log_file
from learntools.kt.data import prepare_data, cv_split
import learntools.deploy.config as config


@log_me()
def run(task_num, model_type=0, **kwargs):
    if model_type == 0:
        from learntools.kt.deepkt import DeepKT as SelectedModel
    elif model_type == 1:
        from learntools.kt.lrkt import build_model  # TODO: UNBREAK
    elif model_type == 2:
        from learntools.kt.kt2 import build_model  # TODO: UNBREAK
    else:
        raise Exception("model type is not valid")

    prepared_data = prepare_data(**kwargs)
    train_idx, valid_idx = cv_split(prepared_data, fold_index=task_num, **kwargs)

    model = SelectedModel((prepared_data, train_idx, valid_idx), **kwargs)
    model.train_full()


if __name__ == '__main__':
    default_dataset = 'data/data5.gz'

    parser = argparse.ArgumentParser(description="run an experiment on this computer")
    parser.add_argument('-p', dest='param_set', type=str, default='default',
                        choices=config.all_param_set_keys,
                        help='the name of the parameter set that we want to use')
    parser.add_argument('--f', dest='file', type=str, default=None,
                        help='the data file to use')
    parser.add_argument('-o', dest='outname', type=str, default=gen_log_name(),
                        help='name for the log file to be generated')
    parser.add_argument('-tn', dest='task_num', type=int, default=0,
                        help='a way to separate different runs of the same parameter-set')
    args = parser.parse_args()

    params = config.get_config(args.param_set)
    set_log_file(args.outname)
    if args.file:
        params['dataset_name'] = args.file
    elif 'dataset_name' not in params:
        params['dataset_name'] = default_dataset
    run(0, **params)
    print "finished"
    if sys.platform.startswith('win'):
        from win_utils import winalert
        winalert()
