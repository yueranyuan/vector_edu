from itertools import izip, ifilter
import csv

import numpy as np

from learntools.emotiv.data import prepare_data
from learntools.data import Dataset


def test_prepare_data():
    dataset_name = 'raw_data/all_seigel.txt'
    data = prepare_data(dataset_name)
    with open(dataset_name, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        reader.next()
        for data_row, csv_row in izip(data.orig, reader):
            strs1 = data_row[:2]
            strs2 = csv_row[:2]
            eegs1 = np.asarray(data_row[2:-1])
            eegs2 = np.asarray([float(e) for e in csv_row[2:]])
            assert strs1 == strs2
            assert np.allclose(eegs1, eegs2)


def test_prepare_data_select_cond():
    dataset_name = 'raw_data/all_seigel.txt'
    conds = ['EyesOpen', 'EyesClosed']
    data = prepare_data(dataset_name, conds=conds)
    with open(dataset_name, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        reader.next()
        csv_only_selected = ifilter(lambda row: row[1] in conds, reader)
        for data_row, csv_row in izip(data.orig, csv_only_selected):
            strs1 = data_row[:2]
            strs2 = csv_row[:2]
            eegs1 = np.asarray(data_row[2:-1])
            eegs2 = np.asarray([float(e) for e in csv_row[2:]])
            assert strs1 == strs2
            assert np.allclose(eegs1, eegs2)
