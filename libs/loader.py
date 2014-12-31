from __future__ import division
import numpy as np
import csv
from time import mktime
from datetime import datetime
from itertools import chain, starmap, imap, izip
from collections import namedtuple

from utils import flatten, transpose

LISTEN_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


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


def parse_time(time_str, form=LISTEN_TIME_FORMAT):
    t = datetime.strptime(time_str, form)
    return int(round((mktime(t.timetuple()) + t.microsecond / 1000000) * 100))


def format_time(time_int, form=LISTEN_TIME_FORMAT):
    seconds = time_int / 100.
    microsecond = (seconds % 1) * 10000
    d = datetime.fromtimestamp(seconds)
    d.replace(microsecond=int(round(microsecond)))
    return datetime.strftime(d, form)


def load(fname, numeric=(), numeric_float=(), time=(), enum=(), text=(),
         time_format=LISTEN_TIME_FORMAT):
    # this ensures that the inputs can be immutable tuples while maintaining
    # mutability inside the function. The extra memory usage is minimal
    numeric = [x for x in numeric]
    numeric_float = [x for x in numeric_float]
    time = [x for x in time]
    enum = [x for x in enum]
    text = [x for x in text]

    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = reader.next()
        # translate string to index for speed
        column_types = [numeric, numeric_float, time, enum, text]
        try:
            for hs in column_types:
                for i, h in enumerate(hs):
                    hs[i] = (header.index(h), h)
        except ValueError:
            raise Exception('header {h} not in file {fname}'.format(h=h, fname=fname))

        # build enum ids and remove all unneeded columns
        rows = []
        columns = list(chain(numeric, numeric_float, time, enum, text))
        enum_dict = {e: {} for i, e in enum}
        for row in reader:
            try:  # empty lines can cause problems
                for i, e in enum:
                    if row[i] not in enum_dict[e]:
                        enum_dict[e][row[i]] = len(enum_dict[e])
                rows.append([row[i] for i, h in columns])
            except IndexError:
                if len(row) != 0:
                    raise  # reraise error if it's not due to empty-line

    # create data structures to hold the loaded data
    def add_type(arr, type_):
        return [(v, type_) for i, v in arr]
    column_dtypes = list(chain.from_iterable(
        starmap(add_type, [(numeric, 'i4'), (numeric_float, 'f4'), (time, 'i8'), (enum, 'i4')])))
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
            m[row_num][i] = parse_time(row[i], time_format)
        for i, n in numeric:
            m[row_num][i] = int(row[i])
        for i, n in numeric_float:
            m[row_num][i] = float(row[i])
        for t_i, (i, n) in enumerate(text):
            text_store[t_i][row_num] = str(row[i])
    return m, enum_dict, text_store


def save(fname, numeric=None, numeric_float=None, enum=None, enum_dict=None,
         time=None, time_format=None, text_store=None, header=None):
    if text_store:
        # TODO: implement text_store
        raise NotImplemented('text_store serialization is not yet implemented')

    # can't use {} as a default param value due to quirk in python
    numeric = numeric or {}
    numeric_float = numeric_float or {}
    enum = enum or {}
    enum_dict = enum_dict or {}
    time = time or {}
    time_format = time_format or {}
    text_store = text_store or {}

    # create iterators for processing each type of data
    def iter_enum(key, data):
        reverse_lookup_table = [None] * len(enum_dict[key])
        for e, i in enum_dict[key].iteritems():
            reverse_lookup_table[i] = e
        for d in data:
            yield reverse_lookup_table[int(d)]

    def iter_time(key, data):
        for d in data:
            yield format_time(d, time_format)

    # apply the appropriate type of processor to each column
    def process(data_dict, to_iter=None):
        if not to_iter:
            iterable_columns = imap(iter, data_dict.itervalues())
        else:
            iterable_columns = starmap(to_iter, data_dict.iteritems())
        return izip(data_dict.keys(), iterable_columns)
    column_type_processor_pair = ((numeric,),
                                  (numeric_float,),
                                  (enum, iter_enum),
                                  (time, iter_time))
    header_column_pairs = list(chain(*starmap(process, column_type_processor_pair)))

    # set headers, reorder columns based on headers
    header_ = [h for h, c in header_column_pairs]
    if not header:
        header = header_
    if sorted(header) != sorted(header_):
        raise Exception("header {header_} from data dictionary differs from the provided header {header}".format(
            header_=header_, header=header))
    header_column_pairs = sorted(header_column_pairs, key=lambda (h, c): header.index(h))
    columns = [c for h, c in header_column_pairs]

    # out to file
    with open(fname, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for row in izip(*columns):
            writer.writerow(row)


if __name__ == "__main__":
    data, enum_dict, _ = load('raw_data/task_large.xls', numeric=['cond'], enum=['subject', 'stim', 'block'],
                              time=['start_time', 'end_time'], text=['latency'])
    # a, b, c = load('raw_data/eeg_single.xls', numeric=['sigqual'], enum=['subject'],
    #               time=['start_time', 'end_time'], text=['rawwave'])
    save('asdf.xls', numeric={'cond': data['cond']}, enum={'subject': data['subject']}, enum_dict=enum_dict,
         time={'start_time': data['start_time'], 'end_time': data['end_time']})
    # save('asdf.xls', numeric={'a': [1, 2, 3]}, enum={'b': [0, 0, 1]}, enum_dict={'b': {'x': 0, 'y': 1}}, header=['b', 'a'])
