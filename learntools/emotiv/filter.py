import os

import numpy as np

from learntools.data import Dataset


DEFAULT_CALIBRATION_FILE_LOCATION = 'raw_data/allcalibqualityreport.csv'


def _load_calibration_data(fname=DEFAULT_CALIBRATION_FILE_LOCATION):
    headers = [('dirnum', Dataset.INT), ('fname', Dataset.STR)]
    num_headers = ('numbadchannels', 'numinterpchannels', 'numsuspectmean', 'numsuspectstd',
                   'numheadmotioncorr', 'numprocheadmotioncorr')
    headers += [(name, Dataset.INT) for name in num_headers]
    return Dataset.from_csv(fname, headers, delimiter=',')


def _get_acceptable_ids(fname=DEFAULT_CALIBRATION_FILE_LOCATION):
    data = _load_calibration_data(fname)

    def _acceptable(row_i):
        return (
            data['numbadchannels'][row_i] < 4 and
            data['numinterpchannels'][row_i] < 4 and
            data['numsuspectmean'][row_i] < 4 and
            data['numsuspectstd'][row_i] < 4 and
            data['numheadmotioncorr'][row_i] < 4 and
            data['numprocheadmotioncorr'][row_i] < 4
        )

    acceptable_rows = filter(_acceptable, xrange(data.n_rows))
    acceptable_ids = [data['fname'][i] for i in acceptable_rows]
    return acceptable_ids


def filter_data(data, calibration_file=DEFAULT_CALIBRATION_FILE_LOCATION, no_children=True, remove_suffix=False):
    subjects = data.orig['subject']
    subject_ids = [subject.split('_')[0] for subject in subjects]
    if remove_suffix:
        subject_ids = [subject[:-len('.edf')] for subject in subject_ids if subject.endswith('.edf')]
    acceptable_ids = _get_acceptable_ids(fname=calibration_file)
    acceptable_ids = [os.path.splitext(x)[0] for x in acceptable_ids]
    if no_children:
        acceptable_ids = filter(lambda(_id): 'Child' not in _id, acceptable_ids)
    subject_mask = [s in acceptable_ids for s in subject_ids]
    data.mask(subject_mask)
    print("subjects: {}".format(np.unique(data.orig['subject'])))
    return data