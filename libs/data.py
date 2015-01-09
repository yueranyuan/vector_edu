import gzip
import cPickle
from collections import Counter
from itertools import count, izip, groupby
import itertools
import heapq

import numpy


class DataSet:
    def __init__(self):
        self.skills = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.cond = numpy.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])


def gen_data(fname):
    inp = numpy.array([[0], [0], [0], [1], [1], [1]])
    target = numpy.array([1, 1, 1, 0, 0, 0])
    set_ = (inp, target)
    with gzip.open(fname, 'w') as f:
        cPickle.dump((set_, set_, set_), f)


def convert_task_from_xls(fname, outname):
    from loader import load
    data, enum_dict, _ = load(
        fname,
        numeric=['cond'],
        enum=['subject', 'stim', 'block'],
        time=['start_time', 'end_time'])
    stim_pairs = list(enum_dict['stim'].iteritems())
    subject_pairs = list(enum_dict['subject'].iteritems())
    skill = data['stim']
    subject = data['subject']
    correct = data['cond']
    start_time = data['start_time']
    end_time = data['end_time']
    with gzip.open(outname, 'w') as f:
        cPickle.dump((subject, start_time, end_time, skill, correct, subject_pairs, stim_pairs), f)


def convert_eeg_from_xls(fname, outname, cutoffs=(0.5, 4.0, 7.0, 12.0, 30.0)):
    from loader import load
    from eeg import signal_to_freq_bins
    data, enum_dict, text = load(
        fname,
        numeric=['sigqual'],
        enum=['subject'],
        time=['start_time', 'end_time'],
        text=['rawwave'])
    subject_pairs = list(enum_dict['subject'].iteritems())
    subject = data['subject']
    sigqual = data['sigqual']
    start_time = data['start_time']
    end_time = data['end_time']
    cutoffs = list(cutoffs)
    eeg_freq = numpy.empty((len(text.rawwave), len(cutoffs) - 1))
    for i, eeg_str in enumerate(text.rawwave):
        eeg = [float(d) for d in eeg_str.strip().split(' ')]
        eeg_freq[i] = tuple(signal_to_freq_bins(eeg, cutoffs=cutoffs, sampling_rate=512))
    with gzip.open(outname, 'w') as f:
        cPickle.dump((subject, start_time, end_time, sigqual, eeg_freq, subject_pairs), f)


def align_data(task_name, eeg_name, out_name, sigqual_cutoff=200):
    with gzip.open(task_name, 'rb') as task_f, gzip.open(eeg_name, 'rb') as eeg_f:
        task_subject, task_start, task_end, skill, correct, task_subject_pairs, stim_pairs = cPickle.load(task_f)
        eeg_subject, eeg_start, eeg_end, sigqual, eeg_freq, eeg_subject_pairs = cPickle.load(eeg_f)
    eeg_freq = numpy.asarray(eeg_freq, dtype='float32')
    num_tasks = len(task_start)

    # Step1: convert to dictionary with subject_id as keys and rows sorted by
    # start_time as values
    def convert_format(subject, start, *rest, **kwargs):
        data_sorted = sorted(izip(subject, start, *rest), key=lambda v: v[:2])
        data_by_subject = {k: list(v) for k, v in groupby(data_sorted, lambda v: v[0])}
        if 'subject_pairs' in kwargs:
            subject_dict = {k: v for v, k in kwargs['subject_pairs']}
            data_by_subject = {subject_dict[k]: v for k, v in data_by_subject.iteritems()}
        return data_by_subject
    task_by_subject = convert_format(task_subject, task_start, task_end, count(0),
                                     subject_pairs=task_subject_pairs)
    eeg_by_subject = convert_format(eeg_subject, eeg_start, eeg_end, count(0),
                                    subject_pairs=eeg_subject_pairs)

    # Step2: efficiently create mapping between task and eeg using the structured data
    task_eeg_mapping = [None] * num_tasks
    for sub, task in task_by_subject.iteritems():
        if sub not in eeg_by_subject:
            continue
        eeg = eeg_by_subject[sub]
        num_sub_eegs = len(eeg)
        eeg_pointer = 0
        try:
            for t in task:
                _, t_start, t_end, t_i = t
                # throw away eeg before the current task
                # this works because the tasks are sorted by start_time
                while eeg_pointer < num_sub_eegs and eeg[eeg_pointer][2] < t_start:
                    eeg_pointer += 1
                    if eeg_pointer > num_sub_eegs:
                        raise StopIteration
                # map eeg onto the current task
                temp_pointer = eeg_pointer
                task_eeg = []
                # TODO: refactor this while loop into a itertools.takewhile
                while temp_pointer < num_sub_eegs and eeg[temp_pointer][1] < t_end:
                    eeg_i = eeg[temp_pointer][3]
                    if sigqual[eeg_i] < sigqual_cutoff:
                        task_eeg.append(eeg_i)
                    temp_pointer += 1
                if task_eeg:
                    task_eeg_mapping[t_i] = task_eeg
        except StopIteration:
            pass

    # Step3: compute eeg features for each task based on aligned eeg
    def compute_eeg_features(eeg_idxs):
        if eeg_idxs is None:
            return None
        return numpy.mean(eeg_freq[eeg_idxs], axis=0)
    features = [compute_eeg_features(ei) if ei else None for ei in task_eeg_mapping]

    # Step4: write data file for use by classifier
    with gzip.open(out_name, 'w') as f:
        cPickle.dump((task_subject, skill, correct, task_start, features, stim_pairs), f)


def _get_ngrams(stims, pairs):
    word_id_count = Counter(stims)
    ngrams = Counter()
    for (word, word_id) in pairs:
        cnt = word_id_count[word_id]
        for l in word:
            ngrams[l] += cnt
        buffered_word = '^{word}$'.format(word=word)
        for b1, b2 in zip(buffered_word, buffered_word[1:]):
            ngrams[b1 + b2] += cnt
    # remove non-letter keys in-place
    bad_chars = set('()1234567890')
    bad_keys = itertools.ifilter(lambda k: set(k) & bad_chars, ngrams.iterkeys())
    for bk in list(bad_keys):  # cast because it seems the reference to bk gets deleted with 'del'
        del ngrams[bk]
    return ngrams


def gen_word_matrix(stims, pairs, vector_length=100):
    stims = stims.flatten()
    ngrams = _get_ngrams(stims, pairs)
    top_pairs = heapq.nlargest(vector_length, ngrams.iteritems(), key=lambda (k, v): v)
    vector_keys = [k for (k, v) in top_pairs]
    padding = [0] * (vector_length - len(top_pairs))
    ordered_pairs = sorted(pairs, key=lambda (w, i): i)
    buffered_words = itertools.imap(lambda (w, i): '^{word}$'.format(word=w), ordered_pairs)
    word_vectors = [[1 if ngram in word else 0 for ngram in vector_keys] + padding
                    for word in buffered_words]
    word_vectors = numpy.asarray(word_vectors)
    return word_vectors


if __name__ == "__main__":
    task_name, eeg_name = 'data/task_data4.gz', 'data/eeg_data4.gz'
    convert_task_from_xls('raw_data/task_large.xls', task_name)
    convert_eeg_from_xls('raw_data/eeg_data_thinkgear_2013_2014.xls', eeg_name)
    align_data('data/task_data4.gz', 'data/eeg_data4.gz', 'data/data4.gz')
