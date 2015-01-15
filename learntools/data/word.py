from itertools import ifilter, imap
from collections import Counter
import heapq

import numpy as np


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
    bad_keys = ifilter(lambda k: set(k) & bad_chars, ngrams.iterkeys())
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
    buffered_words = imap(lambda (w, i): '^{word}$'.format(word=w), ordered_pairs)
    word_vectors = [[1 if ngram in word else 0 for ngram in vector_keys] + padding
                    for word in buffered_words]
    word_vectors = np.asarray(word_vectors)
    return word_vectors
