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


class OriginalColumnView(object):
    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, key):
        return self.owner.to_original(self.owner[key])


class Column(object):
    def __init__(self, name, data=None, **kwargs):
        self.name = name
        self.initialize_data(data, **kwargs)
        self.orig = OriginalColumnView(self)

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

    def to_original(self, value):
        return value

    @property
    def data(self):
        return self[:]

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
    def __init__(self, name, form=LISTEN_TIME_FORMAT, *args, **kwargs):
        super(TimeColumn, self).__init__(name, dtype='i8', *args, **kwargs)
        self.form = form

    def __setitem__(self, key, time_strs):
        if hasattr(time_strs, '__iter__') and not isinstance(time_strs, str):
            value = [parse_time(t, self.form) for t in time_strs]
        else:
            value = parse_time(time_strs, self.form)
        return super(TimeColumn, self).__setitem__(key, value)

    def __getitem__(self, key):
        return super(TimeColumn, self).__getitem__(key)

    def to_original(self, value):
        if hasattr(value, '__iter__') and not isinstance(value, str):
            value_ = [format_time(t, self.form) for t in value]
        else:
            value_ = format_time(value, self.form)
        return value_


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
    def __init__(self, name, enum_dict=None, *args, **kwargs):
        super(EnumColumn, self).__init__(name, dtype='i4', *args, **kwargs)
        self._enum_dict = {} if enum_dict is None else enum_dict
        self._enum_dict_reverse = None

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
        return super(EnumColumn, self).__getitem__(key)

    def to_original(self, value):
        if self._enum_dict_reverse is None:
            self._enum_dict_reverse = dict((v, k) for (k, v) in self._enum_dict.iteritems())
        if hasattr(value, '__iter__') and not isinstance(value, str):
            value_ = [self._enum_dict_reverse[t] for t in value]
        else:
            value_ = self._enum_dict_reverse[value]
        return value_


class OriginalDatasetView(object):
    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, key):
        # key is a column header
        if isinstance(key, str):
            return self.owner.get_column(key).orig[:]
        # key is a row number
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        return [c.orig[key] for c in self.owner.columns]


# TODO: make Dataset resizable (i.e. we need resize to propogate to all columns)
class Dataset(object):
    '''Structured dataset for storing and structuring data to be processed

    The dataset is a collection of columns which all have the same length. The dataset
    enforces the abstraction of the row. This allows the columns to be masked and reordered
    in sync with each other. Once the datatype of the dataset columns are specified,
    the rows can be loaded, written, and manipulated according to that datatype.

    The available Datatypes are: ENUM, TIME, INT, LONG, FLOAT, STR, OBJ,
        MATINT, MATFLOAT

    Datasets emulate a python collection object (alternatively a dictionary and a list).
        Where data = Dataset(*args)
        Retrieving a row: data[row_idx]
        Retrieving a column object: data[column_name]
        Retrieving a cell: data[column_name][row_idx]
        Retrieving the column data (stored inside the column object): data.get_data(column_name)

    Datasets perform transformations on inputs depending on the type of the column.
    For instance, ENUM type columns take a string input but that string is converted into
    a virtual enum. e.g. ['Jan', 'Feb', 'Mar', 'Jan', 'Mar'] will be converted to [0, 1, 2, 0, 2].
    The original input can be retrieved using the 'orig' attribute. The 'orig' attribute is
    merely a view and does not replicate the data.
        Where data = Dataset(*args)
        Retrieving original row: data.orig[row_idx]
        Retrieving original column: data.orig[column_name]
        Retrieving original cell: data.orig[column_name][row_idx]

    Attributes:
        orig (OriginalDatasetView): a view that provides the original form of the data in the Dataset.
            see Dataset's docstring.
    '''
    ENUM = 0
    TIME = 1
    INT = 2
    LONG = 3
    FLOAT = 4
    STR = 5
    OBJ = 6
    MATINT = 7
    MATFLOAT = 8

    def __init__(self, headers, n_rows, form=LISTEN_TIME_FORMAT):
        '''
        Args:
            headers ((string, int)[]): a list of tuples consisting of the name and
                datatype of each column. e.g. [("column1", Dataset.TIME), ("column2", Dataset.INT)]
            n_rows (int): the number of rows the dataset should have. Because
                pre-allocation saves computation time, we try to always know the size of the dataset
                ahead of time. Resizing is currently not supported but is planned.
            form (string, optional): the format string for time strings given to Dataset.TIME type columns
        '''
        self.time_form = form
        self.n_rows = n_rows

        self.header_idx_mapping = {}
        self.columns = []
        self.headers = []
        for h, t in headers:
            self.set_column(h, t)

        self.orig = OriginalDatasetView(self)

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

    def get_data(self, key):
        '''retrieve the raw data of the column not wrapped by the column object.
        This data should NOT be altered.

        Args:
            key (string): column name
        Returns:
            (list): the data stored in the column
        '''
        return self.get_column(key).data

    def get_column(self, key):
        return self.columns[self.header_idx_mapping[key]]

    def set_column(self, header, ctype):
        '''sets or resets a column

        adds a new column of a certain name and type or resets an existing column,
        deleting its data and changing its ctype.

        Args:
            header (string): name of the new column or the name of the existing column to reset
            ctype (int): the type for the column
        '''
        col = self._make_column(header, ctype)

        col_idx = self.header_idx_mapping.get(header, None)
        if col_idx is None:
            self.headers.append((header, ctype))
            self.header_idx_mapping[header] = len(self.headers) - 1
            self.columns.append(col)
        else:
            self.columns[col_idx] = col

    def rename_column(self, key, new_key):
        '''renames an existing column

        Args:
            key (string): current name of the column
            new_key (string): new name for the column
        '''
        idx = self.header_idx_mapping[key]
        self.columns[idx].name = new_key
        self.headers[idx] = (new_key, self.headers[idx][1])
        self.header_idx_mapping.pop(key)
        self.header_idx_mapping[new_key] = idx

    def _resize(self, n_rows):
        self.n_rows = n_rows

    def reorder(self, order_i):
        '''reorder the rows.

        Args:
            order_i (int[]): the value of each index in this list represents the row number of
                the row to be moved to that index. The data [2, 4, 6, 8, 10] reorded with [2, 1, 4, 0]
                would be [6, 4, 10, 2]
        '''
        for c in self.columns:
            if isinstance(c.data, np.ndarray):
                c2 = c[order_i]
            else:
                c2 = [c[i] for i in order_i]
            c.data = c2
        self._resize(len(order_i))

    def mask(self, mask_i):
        '''mask rows. Masked rows are permanently removed

        Args:
            mask_i (bool[]): the boolean in each position in the mask represents
                a row. Masking [1, 2, 3, 4, 5] with [True, True, False, False, True]
                results in [1, 2, 5]
        '''
        mask_i = [bool(i) for i in mask_i]
        for c in self.columns:
            if isinstance(c.data, np.ndarray):
                c2 = c[[i for i in xrange(len(mask_i)) if mask_i[i]]]
            else:
                c2 = list(compress(c.data, mask_i))
            c.data = c2
        self._resize(sum(mask_i))

    def __setitem__(self, key, values):
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        for c, v in izip(self.columns, values):
            c[key] = v

    def __getitem__(self, key):
        # key is a column header
        if isinstance(key, str):
            return self.get_column(key)
        # key is a row number
        if not isinstance(key, int):
            raise Exception("only integer keys can be used for datasets (sorry)")
        return [c[key] for c in self.columns]

    def __len__(self):
        return self.n_rows

    def __str__(self):
        return "<Dataset[" + ', '.join([c.name for c in self.columns]) + "]>"

    def to_pickle(self):
        '''convert the dataset into a serializable format

        Retruns:
            (tuple): serializable tuple representing the data stored in the dataset
        '''
        n_rows = len(self.columns[0])
        headers = self.headers
        time_form = self.time_form

        # get data back in it's former format so it can be read back in the same way
        # TODO: more efficient serialization and reloading to save processing at least for some columns
        data = zip(*[col.orig for col in self.columns])
        return (headers, n_rows, time_form, data)

    @classmethod
    def from_pickle(cls, (headers, n_rows, time_form, data)):
        '''load the data from a serializable format

        Args:
            (tuple): tuple representing the data stored in the dataset

        Returns:
            (Dataset): the loaded Dataset object
        '''
        dataset = cls(headers, n_rows=n_rows, form=time_form)
        for i, row in enumerate(data):
            dataset[i] = data[i]
        return dataset

    @classmethod
    def from_csv(cls, fname, headers, delimiter='\t', **kwargs):
        '''load a dataset from a csv file

        Args:
            fname (string): the location of the csv file
            headers ((string, int)[]) a list of tuples consisting of the name and
                datatype of each column. e.g. [("column1", Dataset.TIME), ("column2", Dataset.INT)]
            delimiters (char): the delimiter of the csv file

        Returns:
            (Dataset): the loaded Dataset object
        '''
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


def _cv_split_helper(splits, cv_fold=0, percent=None):
    if percent is not None:
        from math import ceil
        n_heldout = int(ceil(len(splits) * percent))
        heldout = splits[(cv_fold * n_heldout):((cv_fold + 1) * n_heldout)]
    else:
        heldout = [splits[cv_fold % len(splits)]]
    return heldout


def cv_split(ds, cv_fold=0, split_on=None, percent=None, **kwargs):
    # cross-validation split
    if split_on:
        splits = np.unique(ds[split_on])
        heldout = _cv_split_helper(splits, cv_fold=cv_fold, percent=percent)

        from operator import or_
        mask = reduce(or_, imap(lambda s: np.equal(ds[split_on], s), heldout))
        train_idx = np.nonzero(np.logical_not(mask))[0]
        valid_idx = np.nonzero(mask)[0]
    else:
        from learntools.libs.utils import idx_to_mask, mask_to_idx
        heldout = _cv_split_helper(range(ds.n_rows), cv_fold=cv_fold, percent=percent)
        valid_idx = heldout
        train_mask = np.logical_not(idx_to_mask(valid_idx, mask_len=ds.n_rows))
        train_idx = mask_to_idx(train_mask)

    # print/log what we held out
    split_on_str = split_on if split_on else 'index'
    info = '{split_on} {heldout} are held out'.format(split_on=split_on_str, heldout=heldout)
    from learntools.libs.logger import log
    try:
        log(info, True)
    except:
        print '[was not logged] {}'.format(info)

    return train_idx, valid_idx
