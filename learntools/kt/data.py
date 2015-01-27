import gzip
import cPickle
from itertools import groupby, compress, imap
from operator import or_

import numpy as np

from learntools.data import Dataset
from learntools.libs.logger import log, log_me
from learntools.libs.utils import normalize_table


def convert_task_from_xls(fname, outname=None):
    headers = (('subject', Dataset.ENUM),
               ('stim', Dataset.ENUM),
               ('block', Dataset.ENUM),
               ('start_time', Dataset.TIME),
               ('end_time', Dataset.TIME))
    data = Dataset.from_csv(fname, headers)

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
    data = Dataset.from_csv(fname, headers)
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
    # Step1: convert to dictionary with subject_id as keys and rows sorted by
    # start_time as values
    if isinstance(task_data, str):
        with gzip.open(task_data, 'rb') as f:
            task_data = Dataset.from_pickle(cPickle.load(f))
    if isinstance(eeg_data, str):
        with gzip.open(eeg_data, 'rb') as f:
            eeg_data = Dataset.from_pickle(cPickle.load(f))

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
    task_data.set_column('eeg', Dataset.MATFLOAT)
    for i, ei in enumerate(itask_eeg_mapping):
        features = np.mean(eeg_data['eeg'][ei], axis=0)
        task_data.get_column('eeg')[i] = features
    # Step4: write data file for use by classifier
    if out_name is not None:
        with gzip.open(out_name, 'w') as f:
            cPickle.dump(task_data.to_pickle(), f)
    else:
        return task_data


def cv_split(ds, cv_fold=0, no_new_skills=False, percent=None, **kwargs):
    from learntools.data import cv_split as general_cv_split
    train_idx, valid_idx = general_cv_split(ds,
                                            split_on='subject',
                                            cv_fold=cv_fold,
                                            percent=percent)

    if no_new_skills:
        valid_skill_set = set(ds['skill'][valid_idx])
        train_skill_set = set(ds['skill'][train_idx])
        new_skills = valid_skill_set - train_skill_set
        valid_skill_arr = ds['skill'][valid_idx]
        new_skill_mask = reduce(or_, imap(lambda s: np.equal(valid_skill_arr, s), new_skills))
        valid_idx = valid_idx[np.logical_not(new_skill_mask)]
        log('{} rows are removed because they only occur in the validation set'.format(sum(new_skill_mask)), True)

    return train_idx, valid_idx


@log_me('...loading data')
def prepare_data(dataset_name, top_eeg_n=0, top_n=0, **kwargs):
    from learntools.data import Dataset
    with gzip.open(dataset_name, 'rb') as f:
        ds = Dataset.from_pickle(cPickle.load(f))
    ds.rename_column('stim', 'skill')
    ds.rename_column('cond', 'correct')
    subjects = np.unique(ds['subject'])

    sorted_i = sorted(range(ds.n_rows), key=lambda i: ds['start_time'][i])
    ds.reorder(sorted_i)

    def row_count(subj):
        return sum(np.equal(ds['subject'], subj))
    top_n = top_n or top_eeg_n  # TODO: remove "top_eeg_n" as a config

    if top_n:
        subjects = sorted(subjects, key=row_count)[-top_n:]
    subject_mask = reduce(or_, imap(lambda s: np.equal(ds['subject'], s), subjects))
    ds.mask(subject_mask)
    ds.get_column('eeg').data = normalize_table(ds['eeg'])

    return ds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="generate kt data from raw task and eeg files")
    parser.add_argument('-t', type=str, dest='task', default='raw_data/task_large.xls',
                        help='location of the task file')
    parser.add_argument('-e', type=str, dest='eeg', default='raw_data/eeg_data_thinkgear_2013_2014.xls',
                        help='location of the eeg file')
    parser.add_argument('outfile', type=str, nargs='*', default='data/data5.gz',
                        help='where to store the output file')
    args = parser.parse_args()

    task = convert_task_from_xls(args.task)
    eeg = convert_eeg_from_xls(args.eeg)
    align_data(task, eeg, args.outfile)
