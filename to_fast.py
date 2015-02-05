import os
import csv

from learntools.kt.data import prepare_data, cv_split
from learntools.libs.logger import temp_log
from learntools.data import gen_word_matrix


def _to_fast_with_idxs(fname, headers, skills, data, idxs, single_skill=False, shared_parameters=False):
    with open(fname, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        for i in idxs:
            student = data['subject'][i]
            outcome = 'correct' if data['correct'][i] == 2 else 'incorrect'
            KCs = 'word' if single_skill else data.orig['skill'][i]
            word_feats = skills[data['skill'][i]]
            eeg_feats = data['eeg'][i]
            if shared_parameters:
                student = '{student}_{KCs}'.format(student=student, KCs=KCs)
                KCs = 'word'
            row = [student, outcome, KCs] + list(word_feats) + list(eeg_feats)
            writer.writerow(row)


def to_fast(data, train_fname=None, valid_fname=None, fold=0, vector_length=150, single_skill=False, shared_parameters=False):
    fast_dir = 'fast'
    train_fname = train_fname or os.path.join(fast_dir, 'FAST+deepkt_train{}.txt'.format(fold))
    valid_fname = valid_fname or os.path.join(fast_dir, 'FAST+deepkt_test{}.txt'.format(fold))
    train_idx, valid_idx = cv_split(data, fold_index=fold, no_new_skills=True)
    skills = gen_word_matrix(data.get_data('skill'), data['skill'].enum_pairs, vector_length=vector_length)

    # create fast header
    feat_headers = ['*features_{}'.format(i) for i in xrange(vector_length + data['eeg'].width)]
    headers = ['student', 'outcome', 'KCs'] + feat_headers

    _to_fast_with_idxs(train_fname, headers, skills, data, train_idx, single_skill=single_skill, shared_parameters=shared_parameters)
    _to_fast_with_idxs(valid_fname, headers, skills, data, valid_idx, single_skill=single_skill, shared_parameters=shared_parameters)


@temp_log
def main(folds=None):
    folds = folds or xrange(14)
    data = prepare_data(dataset_name='data/data5.gz', top_n=14)
    for fold in folds:
        to_fast(data, fold=fold, single_skill=False, shared_parameters=False)


if __name__ == "__main__":
    main()