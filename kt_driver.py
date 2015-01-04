import sys
import time
import inspect
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from libs.logger import gen_log_name, log, log_args, set_log_file
import config


def run(**kwargs):
    log_args(inspect.currentframe())

    import kt.data
    import kt.train
    import kt.olddeepkt
    prepared_data = kt.data.prepare_data(**kwargs)

    f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval = (
        kt.olddeepkt.build_model(prepared_data, **kwargs))

    start_time = time.clock()
    best_validation_loss, best_epoch = (
        kt.train.train_model(f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval,
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
    args = parser.parse_args()

    params = config.get_config(args.param_set)
    set_log_file(args.outname)
    if args.file:
        params['dataset_name'] = args.file
    elif 'dataset_name' not in params:
        params['dataset_name'] = default_dataset
    log(run(**params))
    print "finished"
    if sys.platform.startswith('win'):
        from win_utils import winalert
        winalert()
