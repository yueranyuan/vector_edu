from random import random
from collections import namedtuple
from math import log as ln
from math import exp
from libs.utils import combine_dict

LOG_SCALE = 1
LINEAR_SCALE = 2
NORMAL = 1
UNIFORM = 2

Var = namedtuple("Var", ['low', 'high', 'scale', 'dist', 'type'])


def GenVar(low, high=None, scale=LINEAR_SCALE, dist=UNIFORM, type=float):
    if high is None:
        high = low
    return Var(low, high, scale, dist, type)


def get_config(param_set='default'):
    ALL_PARAMS[param_set]

    def instance_var(var):
        if var.scale == LOG_SCALE:
            low, high = ln(var.low), ln(var.high)  # to log scale
        else:
            low, high = var.low, var.high

        if var.dist == UNIFORM:
            val = low + random() * (high - low)
        else:
            raise NotImplementedError(
                "failure on probability distribution {d}".format(d=var.dist))

        if var.scale == LOG_SCALE:
            val = exp(val)  # reverse log
        if var.type == int:
            val += 0.5
        return var.type(val)

    return {n: instance_var(v) for n, v in ALL_PARAMS[param_set].iteritems()}


##########################
# Dict of All Params #####
##########################

ALL_PARAMS = {}
ALL_PARAMS['default'] = {}
ALL_PARAMS['t1v1'] = {
    'learning_rate': GenVar(0.005, 0.02, scale=LOG_SCALE),
    'L1_reg': GenVar(0.000, 0.0001),
    'L2_reg': GenVar(0.00005, 0.0002),
    'main_net_width': GenVar(300, 700, type=int),
    'main_net_depth': GenVar(1, 2, type=int),
    'combiner_width': GenVar(100, 300, type=int),
    'combiner_depth': GenVar(1, 2, type=int),
    'skill_vector_len': GenVar(50, 200, type=int),
    'dropout_p': GenVar(0, 0.4)
}
ALL_PARAMS['eeg1'] = combine_dict(ALL_PARAMS['t1v1'], {'top_eeg_n': GenVar(14, type=int)})
ALL_PARAMS['tiny'] = {
    'n_epochs': GenVar(2)
}
ALL_PARAMS['eeg_off'] = {'previous_eeg_on': GenVar(0, type=int),
                         'current_eeg_on': GenVar(0, type=int)}
ALL_PARAMS['t1v1small'] = combine_dict(ALL_PARAMS['t1v1'], {'n_epochs': GenVar(12, type=int)})


all_param_set_keys = ALL_PARAMS.keys()

if __name__ == '__main__':
    print get_config('t1v1')
