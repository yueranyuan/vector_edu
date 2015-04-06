from random import random
from collections import namedtuple
from math import log as ln
from math import exp

from learntools.libs.utils import combine_dict

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
        if type(var) in (int, long, float, str):
            return var
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

# legacy configurations
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
ALL_PARAMS['t1v1small'] = combine_dict(ALL_PARAMS['t1v1'], {'n_epochs': GenVar(12, type=int)})
ALL_PARAMS['medium_batch'] = {'batch_size': GenVar(90, type=int)}
ALL_PARAMS['big_batch2'] = {'batch_size': GenVar(150, type=int)}
ALL_PARAMS['big_batch3'] = {'batch_size': GenVar(180, type=int)}

# current configurations
ALL_PARAMS['default'] = {}
ALL_PARAMS['basic'] = ALL_PARAMS['t1v1']
ALL_PARAMS['basic_eeg'] = ALL_PARAMS['eeg1']
ALL_PARAMS['eeg_toggle'] = {'previous_eeg_on': GenVar(0, 1, type=int),
                            'current_eeg_on': GenVar(0, 1, type=int)}
ALL_PARAMS['basic_eeg_toggle'] = combine_dict(ALL_PARAMS['basic_eeg'],
                                              ALL_PARAMS['eeg_toggle'])
ALL_PARAMS['eeg_off'] = {'previous_eeg_on': GenVar(0, type=int),
                         'current_eeg_on': GenVar(0, type=int)}
ALL_PARAMS['basic_eeg_off'] = combine_dict(ALL_PARAMS['basic_eeg'], ALL_PARAMS['eeg_off'])
ALL_PARAMS['big_batch'] = {'batch_size': GenVar(120, type=int)}
ALL_PARAMS['basic_eeg_toggle_big_batch'] = combine_dict(ALL_PARAMS['basic_eeg_toggle'],
                                                        ALL_PARAMS['big_batch'])
ALL_PARAMS['mutable'] = {'mutable_skill': GenVar(0, 1, type=int)}
ALL_PARAMS['fix1'] = {'L1_reg': GenVar(0., 0.000005),
                      'main_net_width': GenVar(500, 700, type=int),
                      'combiner_depth': GenVar(1, type=int)}
ALL_PARAMS['config1'] = combine_dict(ALL_PARAMS['basic_eeg_toggle'],
                                     ALL_PARAMS['mutable'],
                                     ALL_PARAMS['fix1'])
ALL_PARAMS['eeg2'] = {'previous_eeg_on': GenVar(0, 1, type=int),
                      'current_eeg_on': GenVar(1, type=int),
                      'eeg_only': GenVar(1, type=int)}
ALL_PARAMS['eeg_off2'] = {'previous_eeg_on': GenVar(0, type=int),
                          'current_eeg_on': GenVar(0, type=int),
                          'eeg_only': GenVar(0, type=int)}
ALL_PARAMS['config1_noeeg'] = combine_dict(ALL_PARAMS['config1'],
                                           ALL_PARAMS['eeg_off2'])
ALL_PARAMS['config1_eeg'] = combine_dict(ALL_PARAMS['config1'],
                                         ALL_PARAMS['eeg2'])
ALL_PARAMS['fix2'] = combine_dict(ALL_PARAMS['eeg2'],
                                  {'L2_reg': GenVar(0.00015, 0.0004),
                                   'mutable_skill': GenVar(0, type=int),
                                   'main_net_width': GenVar(470, 630, type=int),
                                   'main_net_depth': GenVar(1, type=int),
                                   'skill_vector_len': GenVar(80, 160, type=int)})
ALL_PARAMS['config2'] = combine_dict(ALL_PARAMS['basic_eeg_toggle'],
                                     ALL_PARAMS['fix1'],
                                     ALL_PARAMS['fix2'])
ALL_PARAMS['fix3'] = {'learning_rate': GenVar(0.012, 0.02),
                      'L2_reg': GenVar(0.0003, 0.0005)}
ALL_PARAMS['config3'] = combine_dict(ALL_PARAMS['basic_eeg_toggle'],
                                     ALL_PARAMS['fix1'],
                                     ALL_PARAMS['fix2'],
                                     ALL_PARAMS['fix3'])
ALL_PARAMS['top12'] = {'top_eeg_n': GenVar(12, type=int)}
ALL_PARAMS['config3_top12'] = combine_dict(ALL_PARAMS['config3'],
                                           ALL_PARAMS['top12'])
ALL_PARAMS['wide_search'] = {'learning_rate': GenVar(0.005, 0.02, scale=LOG_SCALE),
                             'L2_reg': GenVar(0.00005, 0.0003),
                             'main_net_width': GenVar(300, 700, type=int),
                             'combiner_width': GenVar(100, 300, type=int),
                             'skill_vector_len': GenVar(50, 200, type=int),
                             'dropout_p': GenVar(0, 0.4)}
ALL_PARAMS['config4'] = combine_dict(ALL_PARAMS['config3'],
                                     ALL_PARAMS['wide_search'])
ALL_PARAMS['kt_config5'] = combine_dict(ALL_PARAMS['config3'],
                                        {'model_type': GenVar(2)})
ALL_PARAMS['lrkt_config5'] = combine_dict(ALL_PARAMS['config3'],
                                          {'model_type': GenVar(1)})
ALL_PARAMS['deep_config5'] = combine_dict(ALL_PARAMS['config3'],
                                          {'model_type': GenVar(0)})
ALL_PARAMS['eeg_on'] = {'previous_eeg_on': GenVar(1, type=int),
                        'current_eeg_on': GenVar(1, type=int)}
ALL_PARAMS['deep_config5_noeeg'] = combine_dict(ALL_PARAMS['deep_config5'],
                                                ALL_PARAMS['eeg_off'])
ALL_PARAMS['deep_config5_eeg'] = combine_dict(ALL_PARAMS['deep_config5'],
                                              ALL_PARAMS['eeg_on'])
ALL_PARAMS['combiner_off'] = {'combiner_on': GenVar(0, type=int)}
ALL_PARAMS['deep_config5_nocombiner_eeg'] = combine_dict(ALL_PARAMS['deep_config5'],
                                                         ALL_PARAMS['eeg_on'],
                                                         ALL_PARAMS['combiner_off'])
ALL_PARAMS['deep_config5_nocombiner_noeeg'] = combine_dict(ALL_PARAMS['deep_config5'],
                                                           ALL_PARAMS['eeg_off'],
                                                           ALL_PARAMS['combiner_off'])
ALL_PARAMS['emotiv_wide_search'] = {'learning_rate': GenVar(0.001, 0.02, scale=LOG_SCALE),
                                    'L2_reg': GenVar(0.00005, 0.0003),
                                    'classifier_width': GenVar(100, 700, type=int),
                                    'classifier_depth': GenVar(1, 3, type=int),
                                    'dropout_p': GenVar(0, 0.4),
                                    'n_epochs': 4000,
                                    'patience': 4000}
ALL_PARAMS['emotiv_update1'] = {'dropout_p': 0.,
                                'n_epochs': 50000,
                                'patience': 4000,
                                'patience_increase': 2000}
ALL_PARAMS['emotiv_wide_search2'] = combine_dict(ALL_PARAMS['emotiv_wide_search'],
                                                 ALL_PARAMS['emotiv_update1'])
ALL_PARAMS['emotiv_update2'] = {'dropout_p': GenVar(0.0, 0.1),
                                'learning_rate': GenVar(0.001, 0.02, scale=LOG_SCALE),
                                'patience': 1000,
                                'patience_increase': 500,
                                'subject_norm': GenVar(0, 1, type=int),
                                'clip': GenVar(0, 1, type=int),
                                'rand_cv': GenVar(0, 1, type=int),
                                'data_name': GenVar(0, 1, type=int)}
ALL_PARAMS['emotiv_wide_search3'] = combine_dict(ALL_PARAMS['emotiv_wide_search2'],
                                                 ALL_PARAMS['emotiv_update2'])
ALL_PARAMS['fft_search'] = {'duration': GenVar(5, 15, type=int),
                            'fft_window': GenVar(1.0, 3.0)}
ALL_PARAMS['wavelet_search'] = {'wavelet_depth': GenVar(3, 7, type=int),
                                'wavelet_family': GenVar(1, 6, type=int),
                                'duration': GenVar(5, 10, type=int)}
ALL_PARAMS['emotiv_wide_search4'] = combine_dict(ALL_PARAMS['emotiv_wide_search3'],
                                                 ALL_PARAMS['wavelet_search'])
ALL_PARAMS['short_train'] = {'patience': 400,
                             'patience_increase': 300}
ALL_PARAMS['emotiv_wide_search_autoencode'] = combine_dict(ALL_PARAMS['emotiv_wide_search3'],
                                                           ALL_PARAMS['short_train'])
ALL_PARAMS['randomforest'] = {'n_estimators': GenVar(30, 90, type=int),
                              'average_n_predictions': 0}
ALL_PARAMS['knn'] = {'n_neighbors': GenVar(3, 20, type=int)}
all_param_set_keys = ALL_PARAMS.keys()