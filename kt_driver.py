import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from learntools.libs.logger import gen_log_name, log_me, log, set_log_file
from learntools.kt.data import prepare_data
from learntools.model.train import train_model
import config


@log_me()
def run(task_num, model_type=0, **kwargs):
    if model_type == 0:
        from learntools.kt.deepkt import build_model
    elif model_type == 1:
        from learntools.kt.lrkt import build_model
    elif model_type == 2:
        from learntools.kt.kt2 import build_model
    else:
        raise Exception("model type is not valid")

    prepared_data = prepare_data(cv_fold=task_num, **kwargs)

    f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval = (
        build_model(prepared_data, **kwargs))

    start_time = time.clock()
    best_validation_loss, best_epoch = (
        train_model(f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval,
                    **kwargs))
    end_time = time.clock()
    training_time = (end_time - start_time) / 60.

    log(('Optimization complete. Best validation score of %f %%') %
        (best_validation_loss * 100.), True)
    log('Code ran for ran for %.2fm' % (training_time))
    return (best_validation_loss * 100., best_epoch + 1, training_time)


if __name__ == '__main__':
    default_dataset = 'data/data4.gz'

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
    # params['task_num'] = args.task_num
    log(run(0, **params))
    print "finished"
    if sys.platform.startswith('win'):
        from win_utils import winalert
        winalert()
