import gzip
import cPickle
from collections import Counter
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


def convert_from_xls(fname, outname):
    from loader import load
    data, stim_pairs = load(fname,
        numeric=['cond'],
        enum=['subject', 'stim', 'block'],
        time=['start_time', 'end_time'])
    skill = data['stim'][:, None]
    subject = data['subject'][:, None]
    correct = data['cond']
    with gzip.open(outname, 'w') as f:
        cPickle.dump((skill, subject, correct, stim_pairs), f)


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
    ordered_pairs = sorted(pairs, key=lambda (w, i): i)
    buffered_words = itertools.imap(lambda (w, i): '^{word}$'.format(word=w), ordered_pairs)
    word_vectors = [[1 if ngram in word else 0 for ngram in vector_keys]
                    for word in buffered_words]
    return word_vectors


if __name__ == "__main__":
    fname = 'data/task_data3.gz'
    convert_from_xls('raw_data/task_large.xls', fname)
