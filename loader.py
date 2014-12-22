from __future__ import division
import numpy as np
import csv
from time import mktime
from datetime import datetime
from itertools import chain
from collections import namedtuple


class DynamicRecArray(object):
    def __init__(self, dtype, size=10):
        self.dtype = np.dtype(dtype)
        self.length = 0
        self.size = size
        self._data = np.empty(self.size, dtype=self.dtype)

    def __len__(self):
        return self.length

    def append(self, rec):
        if self.length == self.size:
            self.size = int(1.5 * self.size)
            self._data = np.resize(self._data, self.size)
        self._data[self.length] = rec
        self.length += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    @property
    def data(self):
        return self._data[:self.length]


def parse_time(time_str):
    t = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    return int((mktime(t.timetuple()) + t.microsecond / 1000000) * 100)


def load(fname, numeric=[], time=[], enum=[], text=[]):
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = reader.next()
        # translate string to index for speed
        column_types = [numeric, time, enum, text]
        try:
            for hs in column_types:
                for i, h in enumerate(hs):
                    hs[i] = (header.index(h), h)
        except ValueError:
            raise Exception('header {h} not in file {fname}'.format(h=h, fname=fname))

        # build enum ids and remove all unneeded columns
        rows = []
        columns = list(chain(numeric, time, enum, text))
        enum_dict = {e:{} for i, e in enum}
        for row in reader:
            for i, e in enum:
                if row[i] not in enum_dict[e]:
                    enum_dict[e][row[i]] = len(enum_dict[e])
            rows.append([row[i] for i, h in columns])

        # create data structures to hold the loaded data
        def add_type(arr, type_):
            return [(v, type_) for i, v in arr]
        column_dtypes = list(chain(*[add_type(*args) for args in
                   [(numeric, 'i4'), (time, 'i8'), (enum, 'i4')]]))
        m = np.empty(len(rows), dtype=column_dtypes)
        TextStore = namedtuple('Text', [h for i, h in text])
        text_store = TextStore(*[[None] * len(rows) for i in xrange(len(text))])
        # reindex headers because the unneeded columns have been removed
        # we have to do it this way because python doesn't support direct pointer access :(
        j = 0
        for hs in column_types:
            for i, h in enumerate(hs):
                hs[i] = (j, h[1])
                j += 1
        # load data into our structure in the proper format
        for row_num, row in enumerate(rows):
            for i, e in enum:
                m[row_num][i] = enum_dict[e][row[i]]
            for i, t in time:
                m[row_num][i] = parse_time(row[i])
            for i, n in numeric:
                m[row_num][i] = int(row[i])
            for t_i, (i, n) in enumerate(text):
                text_store[t_i][row_num] = str(row[i])
        return m, enum_dict, text_store

if __name__ == "__main__":
    load('raw_data/task_large.xls', numeric=['cond'], enum=['subject', 'stim', 'block'],
        time=['start_time', 'end_time'], text=['latency'])
