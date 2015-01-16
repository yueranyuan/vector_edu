import gzip
import cPickle
from itertools import groupby, compress

import numpy as np

from dataset import load, Dataset


def convert_task_from_xls(fname, outname=None):
    headers = (('cond', Dataset.INT),
               ('subject', Dataset.ENUM),
               ('stim', Dataset.ENUM),
               ('block', Dataset.ENUM),
               ('start_time', Dataset.TIME),
               ('end_time', Dataset.TIME))
    data = load(fname, headers)

    if outname is not None:
        with gzip.open(outname, 'w') as f:
            cPickle.dump(data.to_pickle(), f)
    else:
        return data


def convert_eeg_from_xls(fname, outname=None, cutoffs=(0.5, 4.0, 7.0, 12.0, 30.0)):
    from eeg import signal_to_freq_bins
    headers = (('sigqual', Dataset.INT),
               ('subject', Dataset.ENUM),
               ('start_time', Dataset.TIME),
               ('end_time', Dataset.TIME),
               ('rawwave', Dataset.STR))
    data = load(fname, headers)
    cutoffs = list(cutoffs)
    eeg_freq = np.empty((len(data.get_data('rawwave')), len(cutoffs) - 1))
    for i, eeg_str in enumerate(data['rawwave']):
        eeg = [float(d) for d in eeg_str.strip().split(' ')]
        eeg_freq[i] = tuple(signal_to_freq_bins(eeg, cutoffs=cutoffs, sampling_rate=512))
    data.set_column('eeg', Dataset.MATFLOAT, data=eeg_freq)
    if outname is not None:
        with gzip.open(outname, 'w') as f:
            cPickle.dump(data.to_pickle(), f)
    return data


def align_data(task_data, eeg_data, out_name=None, sigqual_cutoff=200):
    # with gzip.open(task_name, 'rb') as task_f, gzip.open(eeg_name, 'rb') as eeg_f:
    #    task_subject, task_start, task_end, skill, correct, task_subject_pairs, stim_pairs = cPickle.load(task_f)
    #    eeg_subject, eeg_start, eeg_end, sigqual, eeg_freq, eeg_subject_pairs = cPickle.load(eeg_f)

    # Step1: convert to dictionary with subject_id as keys and rows sorted by
    # start_time as values
    def convert_format(dataset):
        sorted_i = sorted(xrange(dataset.n_rows),
                          key=lambda i: (dataset['subject'][i], dataset['start_time'][i]))
        subject_groups = groupby(sorted_i, lambda i: dataset['subject'][i])
        subject_dict = dict((v, k) for (k, v) in dataset.get_column('subject').enum_pairs)
        data_by_subject = {subject_dict[k]: list(v) for k, v in subject_groups}
        return data_by_subject

    task_by_subject = convert_format(task_data)
    eeg_by_subject = convert_format(eeg_data)

    # Step2: efficiently create mapping between task and eeg using the structured data
    task_eeg_mapping = [None] * task_data.n_rows
    for sub, task in task_by_subject.iteritems():
        if sub not in eeg_by_subject:
            continue
        eeg = eeg_by_subject[sub]
        num_sub_eegs = len(eeg)
        e_i = 0
        try:
            for t_i in task:
                t_start = task_data['start_time'][t_i]
                t_end = task_data['end_time'][t_i]
                # throw away eeg before the current task
                # this works because the tasks are sorted by start_time
                while e_i < num_sub_eegs and eeg_data['end_time'][eeg[e_i]] < t_start:
                    e_i += 1
                    if e_i > num_sub_eegs:
                        raise StopIteration
                # map eeg onto the current task
                e_i2 = e_i
                task_eeg = []
                # TODO: refactor this while loop into a itertools.takewhile
                while e_i2 < num_sub_eegs and eeg_data['start_time'][eeg[e_i2]] < t_end:
                    if eeg_data['sigqual'][eeg[e_i2]] < sigqual_cutoff:
                        task_eeg.append(eeg[e_i2])
                    e_i2 += 1
                if task_eeg:
                    task_eeg_mapping[t_i] = task_eeg
        except StopIteration:
            pass

    # Step3: compute eeg features for each task based on aligned eeg
    eeg_mask = [bool(ei) for ei in task_eeg_mapping]
    task_data.mask(eeg_mask)
    itask_eeg_mapping = compress(task_eeg_mapping, eeg_mask)
    task_data.set_column('eeg', Dataset.MATINT)
    for i, ei in enumerate(itask_eeg_mapping):
        features = np.mean(eeg_data['eeg'][ei], axis=0)
        task_data.get_column('eeg')[i] = features

    # Step4: write data file for use by classifier
    if out_name is not None:
        with gzip.open(out_name, 'w') as f:
            cPickle.dump(task_data.to_pickle(), f)
    else:
        return task_data


if __name__ == "__main__":
    task_name, eeg_name = 'data/task_data4.gz', 'data/eeg_data4.gz'
    task = convert_task_from_xls('raw_data/task_large.xls')
    eeg = convert_eeg_from_xls('raw_data/eeg_data_thinkgear_2013_2014.xls')
    align_data(task, eeg, 'data/data5.gz')
