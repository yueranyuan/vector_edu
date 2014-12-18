from __future__ import division
import numpy as np
import csv
import time
import datetime


class DynamicRecArray(object):
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.length = 0
        self.size = 10
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
    t = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    return int((time.mktime(t.timetuple()) + t.microsecond / 1000000) * 100)


def load(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        reader.next()  # pop header
        m = DynamicRecArray([('subject', 'i4'), ('stim', 'i4'), ('block', 'i4'),
                            ('start_time', 'i8'), ('end_time', 'i8'), ('cond', 'i4')])
        subjects = {}
        stims = {}
        blocks = {}
        rows = []
        for r in reader:
            rows.append(r)
            machine, subject, start_time, end_time, stim, block, correct, latency, cond = r
            if subject not in subjects:
                subjects[subject] = len(subjects)
            if stim not in stims:
                stims[stim] = len(stims)
            if block not in blocks:
                blocks[block] = len(blocks)
        for r in rows:
            machine, subject, start_time, end_time, stim, block, correct, latency, cond = r
            m.append((subjects[subject], stims[stim], blocks[block], parse_time(start_time), parse_time(end_time), cond))
        return m.data, list(stims.iteritems())

if __name__ == "__main__":
    print(load('task_large.xls'))
