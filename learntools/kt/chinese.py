from itertools import imap
from operator import or_

import numpy as np

from learntools.data import Dataset


SQL_FORMAT = '%Y-%m-%d %H:%M:%S'


def prepare_data(dataset_name, top_n=0, **kwargs):
    headers = (('Time', Dataset.TIME),
               ('Anon Student Id', Dataset.ENUM),
               ('Problem Name', Dataset.ENUM),
               ('Outcome', Dataset.ENUM))
    data = Dataset.from_csv(dataset_name, headers, form=SQL_FORMAT)
    outcome_mask = [d != '' for d in data.orig['Outcome']]
    data.mask(outcome_mask)
    data.rename_column('Time', 'start_time')
    data.rename_column('Anon Student Id', 'subject')
    data.rename_column('Problem Name', 'skill')

    def row_count(subj):
        return sum(np.equal(data['subject'], subj))
    subjects = np.unique(data['subject'])
    if top_n:
        subjects = sorted(subjects, key=row_count)[-top_n:]
        subject_mask = reduce(or_, imap(lambda s: np.equal(data['subject'], s), subjects))
        data.mask(subject_mask)

    data.set_column('eeg', Dataset.MATFLOAT)
    for i in xrange(data.n_rows):
        data.get_column('eeg')[i] = [1]
    data.set_column('correct', Dataset.INT)
    conds = [1 if d == 'INCORRECT' else 2 for d in data.orig['Outcome']]
    for i in xrange(data.n_rows):
        data.get_column('correct')[i] = conds[i]

    return data


if __name__ == "__main__":
    ds = prepare_data('raw_data/chinese_dictation.txt', top_n=40)
    #for row in ds.orig:
    #    print row
