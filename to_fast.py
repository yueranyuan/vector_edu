import csv

from learntools.kt.data import prepare_data, cv_split
from learntools.libs.logger import set_log_file
from learntools.data import gen_word_matrix


def _to_fast_with_idxs(fname, headers, skills, data, idxs, single_skill=False):
    with open(fname, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        for i in idxs:
            student = data['subject'][i]
            outcome = 'correct' if data['correct'][i] == 2 else 'incorrect'
            KCs = 'word' if single_skill else data.orig['skill'][i]
            word_feats = skills[data['skill'][i]]
            row = [student, outcome, KCs] + list(word_feats)
            writer.writerow(row)


def to_fast(data, train_fname=None, valid_fname=None, fold=0, vector_length=150, single_skill=False):
    train_fname = train_fname or 'FAST+deepkt_train{}.txt'.format(fold)
    valid_fname = valid_fname or 'FAST+deepkt_test{}.txt'.format(fold)
    valid_idx, train_idx = cv_split(data, cv_fold=fold, no_new_skill=True)
    skills = gen_word_matrix(data.get_data('skill'), data['skill'].enum_pairs, vector_length=vector_length)

    # create fast header
    feat_headers = ['*features_{}'.format(i) for i in xrange(vector_length)]
    headers = ['student', 'outcome', 'KCs'] + feat_headers

    _to_fast_with_idxs(train_fname, headers, skills, data, train_idx, single_skill=single_skill)
    _to_fast_with_idxs(valid_fname, headers, skills, data, valid_idx, single_skill=single_skill)

set_log_file('temp.log')
#for fold in xrange(14):
#    to_fast(fold=fold, single_skill=False)
data = prepare_data(dataset_name='data/data5.gz', top_n=14)
to_fast(data, fold=0, single_skill=False)
