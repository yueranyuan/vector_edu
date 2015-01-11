from __future__ import division
import numpy as np
import csv
from time import mktime
from datetime import datetime
from itertools import chain, starmap, imap, izip
from collections import namedtuple

LISTEN_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class DynamicRecArray(object):
    def __init__(self, dtype, size=10):
        self.dtype = np.dtype(dtype)
        self.length = 0
        self.size = size
        self._data = np.zeros(self.size, dtype=self.dtype)

    def __len__(self):
        return max(self.length, self._data)

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
        return self._data[:len(self._data)]


class Column(DynamicRecArray):
    def __init__(self, name, dtype='f4', size=10):
        super(Column, self).__init__(dtype=dtype)
        self.name = name

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, values):
        self._data[key] = values


class NumericColumn(Column):
    def __init__(self, name, set_func=None, *args, **kwargs):
        super(NumericColumn, self).__init__(name, *args, **kwargs)
        if set_func is None:
            self.set_func = lambda x: x
        else:
            self.set_func = set_func

    def __setitem__(self, key, values):
        if hasattr(values, '__iter__') and not isinstance(values, str):
            value = [self.set_func(t) for t in values]
        else:
            value = self.set_func(values)
        return super(NumericColumn, self).__setitem__(key, value)


class TimeColumn(Column):
    NUM = 0
    ORIGINAL = 1

    def __init__(self, name, form=LISTEN_TIME_FORMAT, *args, **kwargs):
        super(TimeColumn, self).__init__(name, dtype='i8', *args, **kwargs)
        self.form = LISTEN_TIME_FORMAT
        self.mode = TimeColumn.NUM

    def __setitem__(self, key, time_strs):
        if hasattr(time_strs, '__iter__') and not isinstance(time_strs, str):
            value = [parse_time(t, self.form) for t in time_strs]
        else:
            value = parse_time(time_strs, self.form)
        return super(TimeColumn, self).__setitem__(key, value)

    def __getitem__(self, key):
        if self.mode == TimeColumn.NUM:
            return super(TimeColumn, self).__getitem__(key)
        else:
            values = super(TimeColumn, self).__getitem__(key)
            if hasattr(values, '__iter__') and not isinstance(values, str):
                values = [format_time(t, self.form) for t in values]
            else:
                values = format_time(values, self.form)
            return values


class EnumColumn(Column):
    NUM = 0
    ORIGINAL = 1

    def __init__(self, name, enum_dict=None, *args, **kwargs):
        super(EnumColumn, self).__init__(name, dtype='i4', *args, **kwargs)
        self._enum_dict = {} if enum_dict is None else enum_dict
        self._enum_dict_reverse = None
        self.mode = EnumColumn.NUM

    def __convert_to_dict__(self, str_value):
        if str_value in self._enum_dict:
            return self._enum_dict[str_value]
        else:
            int_val = len(self._enum_dict)
            self._enum_dict[str_value] = int_val
            return int_val

    def __setitem__(self, key, str_value):
        if hasattr(str_value, '__iter__') and not isinstance(str_value, str):
            value = [self.__convert_to_dict__(t) for t in str_value]
        else:
            value = self.__convert_to_dict__(str_value)
        return super(EnumColumn, self).__setitem__(key, value)

    def __getitem__(self, key):
        if self.mode == EnumColumn.NUM:
            return super(EnumColumn, self).__getitem__(key)
        else:
            value = super(EnumColumn, self).__getitem__(key)
            if self._enum_dict_reverse is None:
                self._enum_dict_reverse = dict((v, k) for (k, v) in self._enum_dict.iteritems())
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value_ = [self._enum_dict_reverse[t] for t in value]
            else:
                value_ = self._enum_dict_reverse[value]
            return value_


# TODO: alter string to integer for numeric columns
class Dataset(object):
    ENUM = 0
    TIME = 1
    INT = 2
    LONG = 3
    FLOAT = 4

    NUM = 0
    ORIGINAL = 1

    def __init__(self, headers, n_rows=10, form=LISTEN_TIME_FORMAT):
        self.columns = {}
        self._mode = Dataset.NUM
        for h, t in headers:
            if t == Dataset.ENUM:
                col = EnumColumn(name=h, size=n_rows)
            elif t == Dataset.TIME:
                col = TimeColumn(name=h, size=n_rows, form=form)
            elif t in (Dataset.INT, Dataset.LONG, Dataset.FLOAT):
                if t == Dataset.INT:
                    col_type = 'i4'
                    func_type = lambda x: int(x)
                elif t == Dataset.LONG:
                    col_type = 'i8'
                    func_type = lambda x: long(x)
                elif t == Dataset.FLOAT:
                    col_type = 'f4'
                    func_type = lambda x: float(x)
                col = NumericColumn(name=h, dtype=col_type, set_func=func_type, size=n_rows)
            self.columns[h] = col

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        for c in self.columns.itervalues():
            c.mode = self._mode

    def __setitem__(self, key, values):
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        for c, v in izip(self.columns.itervalues(), values):
            c[key] = v

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        return [c[key] for c in self.columns.itervalues()]


def parse_time(time_str, form=LISTEN_TIME_FORMAT):
    t = datetime.strptime(time_str, form)
    return long(round((mktime(t.timetuple()) + t.microsecond / 1000000) * 100))


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
