import itertools
from itertools import izip

from learntools.data import Dataset
from learntools.libs.utils import normalize_table


def prepare_data(dataset_name, conds=None, **kwargs):
    """load siegle data into a Dataset

    Args:
        conds (string[], optional): list of conditions that we want the dataset to contain
            e.g. ['EyesClosed', 'EyesOpen']. If not provided, all conditions will be loaded

    Returns:
        Dataset: a dataset with the following columns:
            group: the source of the data (contains descriptives about the subject, etc.)
            condition: the task that the subject was doing while the eeg was being recorded
            eeg: eeg feature vector for each task
    """
    bands = ['theta', 'alpha', 'beta', 'gamma']
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    eeg_headers = [('{0}_{1}'.format(*x), Dataset.FLOAT) for x in
                   itertools.product(bands, channels)]
    headers = [('fname', Dataset.STR), ('Condition', Dataset.ENUM)] + eeg_headers
    data = Dataset.from_csv(dataset_name, headers)
    data.rename_column('fname', 'group')
    data.rename_column('Condition', 'condition')
    data.set_column('eeg', Dataset.MATFLOAT)
    for i, eeg in enumerate(izip(*[data[h] for (h, _) in eeg_headers])):
        data.get_column('eeg')[i] = eeg
    data.get_column('eeg').data = normalize_table(data['eeg'])

    # only keep selected conditions
    # TODO: turn this these temporary mode switches into a context
    if conds is not None:
        all_conds = data.orig['condition']  # TODO: follow up on possible deadlock caused
        # when I put data['condition'] in the filter
        selected = filter(lambda i: all_conds[i] in conds, xrange(data.n_rows))
        data.reorder(selected)
        cond_data = data.orig['condition']
        data.set_column('condition', Dataset.ENUM)  # reset the condition column
        for i, c in enumerate(cond_data):
            data.get_column('condition')[i] = c
    return data
