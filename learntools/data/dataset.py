from __future__ import division
import numpy as np
import csv
from time import mktime
from datetime import datetime
from itertools import chain, starmap, imap, izip, compress

LISTEN_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class DynamicRecArray(object):
    def __init__(self, dtype='i4', size=10):
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


class Column(object):
    def __init__(self, name, data=None, **kwargs):
        self.name = name
        self.initialize_data(data, **kwargs)

    def initialize_data(self, data, dtype='i4', size=10, **kwargs):
        if data is None:
            self._data = np.zeros(size, dtype=dtype)
        else:
            self._data = np.asarray(data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, values):
        self._data[key] = values

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __eq__(self, x):
        return self._data == x

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self[:]  # TODO: follow up on the memory efficiency of doing this

    @data.setter
    def data(self, value):
        self._data = value


class ObjectColumn(Column):
    def initialize_data(self, data, size, **kwargs):
        if data is None:
            self._data = [None] * size
        else:
            self._data = data


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


class MatColumn(Column):
    def initialize_data(self, data, dtype='i4', size=10, **kwargs):
        self.n_rows = size
        self.dtype = dtype
        self._data = None

    def __setitem__(self, key, value):
        if not isinstance(value, (np.ndarray)):
            value = np.array(value)
        # lazily initialize data so that we don't have to pre-specify the dimensions
        if self._data is None:
            n_cols = value.shape[-1]
            self._data = np.zeros((self.n_rows, n_cols), dtype=self.dtype)
        return super(MatColumn, self).__setitem__(key, value)

    def __getitem__(self, key):
        if self._data is None:
            raise Exception("Matrix column '{}' not initialized".format(self.name))
        return super(MatColumn, self).__getitem__(key)


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

    @property
    def ienum_pairs(self):
        return self._enum_dict.iteritems()

    @property
    def enum_pairs(self):
        return list(self._enum_dict.iteritems())

    def __getitem__(self, key):
        if self.mode == EnumColumn.NUM:
            return super(EnumColumn, self).__getitem__(key)
        else:
            value = super(EnumColumn, self).__getitem__(key)
            if self._enum_dict_reverse is None:
                self._enum_dict_reverse = dict((v, k) for (k, v) in self._enum_dict.iteritems())
            if hasattr(value, '__iter__') and not isinstance(value, str):
                print value
                print self._enum_dict_reverse
                value_ = [self._enum_dict_reverse[t] for t in value]
            else:
                value_ = self._enum_dict_reverse[value]
            return value_


# TODO: make Dataset resizable (i.e. we need resize to propogate to all columns)
class Dataset(object):
    ENUM = 0
    TIME = 1
    INT = 2
    LONG = 3
    FLOAT = 4
    STR = 5
    OBJ = 6
    MATINT = 7
    MATFLOAT = 8

    NUM = 0
    ORIGINAL = 1

    def __init__(self, headers, n_rows=10, form=LISTEN_TIME_FORMAT):
        self._mode = Dataset.NUM
        self.time_form = form
        self.n_rows = n_rows

        self.header_idx_mapping = {}
        self.columns = []
        self.headers = []
        for h, t in headers:
            self.set_column(h, t)

    def _make_column(self, h, t):
        if t == Dataset.ENUM:
            return EnumColumn(name=h, size=self.n_rows)
        elif t == Dataset.TIME:
            return TimeColumn(name=h, size=self.n_rows, form=self.time_form)
        elif t == Dataset.STR or t == Dataset.OBJ:
            return ObjectColumn(name=h, size=self.n_rows)
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
            return NumericColumn(name=h, dtype=col_type, set_func=func_type,
                                 size=self.n_rows)
        elif t in (Dataset.MATINT, Dataset.MATFLOAT):
            if t == Dataset.MATINT:
                col_type = 'i4'
            elif t == Dataset.MATFLOAT:
                col_type = 'f4'
            return MatColumn(name=h, dtype=col_type, size=self.n_rows)
        else:
            raise Exception('unknown dataset type for column')

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        for c in self.columns:
            c.mode = self._mode

    def get_data(self, key):
        return self.get_column(key).data

    def get_column(self, key):
        return self.columns[self.header_idx_mapping[key]]

    def set_column(self, header, ctype, data=None):
        col = self._make_column(header, ctype)
        if data is not None:
            col[:len(data)] = data

        col_idx = self.header_idx_mapping.get(header, None)
        if col_idx is None:
            self.headers.append((header, ctype))
            self.header_idx_mapping[header] = len(self.headers) - 1
            self.columns.append(col)
        else:
            self.columns[col_idx] = col

    def rename_column(self, key, new_key):
        idx = self.header_idx_mapping[key]
        self.columns[idx].name = new_key
        self.headers[idx] = (new_key, self.headers[idx][1])
        self.header_idx_mapping.pop(key)
        self.header_idx_mapping[new_key] = idx

    def __setitem__(self, key, values):
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        for c, v in izip(self.columns, values):
            c[key] = v

    def __getitem__(self, key):
        # key is a column header
        if isinstance(key, str):
            return self.get_data(key)
        # key is a row number
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        return [c[key] for c in self.columns]

    def __len__(self):
        return self.n_rows

    def to_pickle(self):
        n_rows = len(self.columns[0])
        headers = self.headers
        time_form = self.time_form

        # get data back in it's former format so it can be read back in the same way
        # TODO: more efficient serialization and reloading to save processing at least for some columns
        old_mode = self.mode
        self.mode = Dataset.ORIGINAL
        data = zip(*[col.data for col in self.columns])
        self.mode = old_mode
        return (headers, n_rows, time_form, data)

    @classmethod
    def from_pickle(cls, (headers, n_rows, time_form, data)):
        dataset = cls(headers, n_rows=n_rows, form=time_form)
        for i, row in enumerate(data):
            dataset[i] = data[i]
        return dataset

    def resize(self, n_rows):
        self.n_rows = n_rows

    def reorder(self, order_i):
        oldmode = self.mode
        self.mode = Dataset.NUM
        for c in self.columns:
            if isinstance(c.data, np.ndarray):
                c2 = c[order_i]
            else:
                c2 = [c[i] for i in order_i]
            c.data = c2
        self.resize(len(order_i))
        self.mode = oldmode  # TODO: use a context for temporary modes

    def mask(self, mask_i):
        oldmode = self.mode
        self.mode = Dataset.NUM
        mask_i = [bool(i) for i in mask_i]
        for c in self.columns:
            if isinstance(c.data, np.ndarray):
                c2 = c[[i for i in xrange(len(mask_i)) if mask_i[i]]]
            else:
                c2 = list(compress(c.data, mask_i))
            c.data = c2
        self.resize(sum(mask_i))
        self.mode = oldmode  # TODO: use a context for temporary modes

    @classmethod
    def from_csv(cls, fname, headers, delimiter='\t', **kwargs):
        from learntools.libs.utils import get_column

        # this is so we can allocate memory ahead of time
        # resizing arrays will be more costly than reading the file twice
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            n_rows = sum(1 for row in reader) - 1  # don't count header

        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            file_headers = reader.next()
            data_headers = get_column(headers, 0)
            header_column_idxs = [file_headers.index(h) for i, h in enumerate(data_headers)]
            dataset = cls(headers, n_rows=n_rows, **kwargs)
            for i, row in enumerate(reader):
                dataset[i] = (row[i] for i in header_column_idxs)
        return dataset


def parse_time(time_str, form=LISTEN_TIME_FORMAT):
    t = datetime.strptime(time_str, form)
    return long(round((mktime(t.timetuple()) + t.microsecond / 1000000) * 100))


def format_time(time_int, form=LISTEN_TIME_FORMAT):
    seconds = time_int / 100.
    microsecond = (seconds % 1) * 10000
    d = datetime.fromtimestamp(seconds)
    d.replace(microsecond=int(round(microsecond)))
    return datetime.strftime(d, form)


def load(*args, **kwargs):
    return Dataset.from_csv(*args, **kwargs)


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
