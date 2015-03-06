import operator
from itertools import chain, imap, ifilterfalse, compress
import math

import numpy as np
import scipy.io


# I should probably split these into separate files but it would kind of be a
# waste of a files right now since they'll probably all be in separate ones
def combine_dict(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# transposes a 2d array of arbitrary elements without numpy
def transpose(arr):
    if not isinstance(arr, list):
        arr = list(arr)
        # TODO: issue warning of a potentially expensive operation
    if len(arr) == 0:
        return []
    width = len(arr[0])
    out = [None] * width
    for i in range(width):
        out[i] = [a[i] for a in arr]
    return out


def get_divisors(n):
    for i in xrange(1, int(n / 2 + 1)):
        if n % i == 0:
            yield i
    yield n


# returns min index and min value
def min_idx(arr):
    return min(enumerate(arr), key=operator.itemgetter(1))


# returns max index and max value
def max_idx(arr):
    return max(enumerate(arr), key=operator.itemgetter(1))


def flatten(arr):
    return list(chain.from_iterable(arr))


def clip_outliers(matrix, method='std', axis=None):
    # take things within 25th-75th percentile, with subsampling for speedup
    size = len(matrix)
    samples = math.log(size) if size > 100 else size
    subsample = matrix[::int(size / samples)]
    sorted_subsample = np.sort(subsample, axis=axis)
    if axis is None:
        N = len(sorted_subsample)
    else:
        N = sorted_subsample.shape[axis]
    iqr = np.take(sorted_subsample, range(int(N * 0.25), int(N * 0.75)), axis=axis)
    if iqr.shape[0] == 1:
        raise Exception("insufficient rows in matrix to get reliable interquartial range")
    mean = np.mean(iqr, axis=axis)
    if method == 'iqr':  # use interquartile range
        lower_bound = iqr[0]
        upper_bound = iqr[-1]
        lo_thresh = mean + 1.5 * (lower_bound - mean)
        hi_thresh = mean + 1.5 * (upper_bound - mean)
    elif method == 'std':  # use standard deviation
        std = np.std(iqr, axis=axis)
        lo_thresh = mean - 3.0 * std
        hi_thresh = mean + 3.0 * std
    else:
        raise ValueError("clipping method unknown")
    return np.minimum(np.maximum(lo_thresh, matrix), hi_thresh)


def normalize_standard(matrix, epsilon=1e-7):
    # subsampling for speedup
    size = len(matrix)
    samples = math.log(size)
    subsample = matrix[::int(size / samples)]
    mean = np.mean(subsample, axis=0)
    std = np.std(subsample, axis=0)

    return (matrix - mean) / (std + epsilon)


def normalize_table(table, clip=False, within_subject=None, axis=None):
    if not isinstance(table, np.ndarray):
        table = np.asarray(table)

    if within_subject:
        subjects = np.unique(within_subject)
        for subject in subjects:
            selected_idxs = list(compress(range(len(within_subject)), within_subject == subject))
            table_s = table[selected_idxs]
            norm_table_s = normalize_table(table_s, clip=clip, axis=axis)
            table[selected_idxs] = norm_table_s
        return table

    if clip:
        table = clip_outliers(table)

    mins = table.min(axis=axis)
    maxs = table.max(axis=axis)
    norm_table = (table - mins) / (maxs - mins)
    if np.any(np.isnan(norm_table)):
        # TODO: issue warning all nan
        print "Warning: normalized table contains nans"
        if not np.any(np.isnan(table)):
            print "Warning: nans were not present in input table"
    return norm_table


# converts an index array into the corresponding mask
# example: [1, 3, 4] -> [False, True, False, True, True]
def idx_to_mask(idxs, mask_len=None):
    if not mask_len:
        mask_len = max(idxs) + 1
    mask = np.array([False] * mask_len)
    mask[idxs] = True
    return mask


def mask_to_idx(mask):
    return np.nonzero(mask)[0]


def iget_column(data, i):
    return imap(operator.itemgetter(i), data)


def get_column(data, i):
    return [d[i] for d in data]


# from http://stackoverflow.com/questions/7008608
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class ArgumentNotSpecifiedIndicator(object):
    """ a singleton object to be used as an argument default to represent that no argument was provided.
    This is used as a default when 'None' could be a possible value for the argument.
    """
    pass

ARGUMENT_NOT_SPECIFIED = ArgumentNotSpecifiedIndicator()  # a singleton object to be used as an argument default


class ExceptionFoundIndicator(object):
    """ a singleton for use by exception_safe_map
    """
    pass

_EXCEPTION_FOUND = ExceptionFoundIndicator()  # singleton for use by exception_safe_map


def exception_safe_map(func, data, exception=Exception, fill=ARGUMENT_NOT_SPECIFIED):
    """ a map function that filters out elements where the mapping function generated an exception

    NOTE: this may result in changed indices and length of input data

    Args:
        func (lambda): the mapping function
        data (list): some iterable to map over
        exception (BaseException, optional): the specific type of exception to make safe, all other exceptions will be
            raised
        fill (object, optional): object to be used in place of the element that generated an exception. If not specified
            then the element will simply be omitted. Note that indices and length may change as a result.

    Returns:
        (list): list mapped with func and with all exception-generating elements removed
    """
    def safe_func(inp):
        try:
            return func(inp)
        except exception:
            return _EXCEPTION_FOUND

    mapped = map(safe_func, data)
    if fill is ARGUMENT_NOT_SPECIFIED:
        mapped = filter(lambda(x): x is not _EXCEPTION_FOUND, mapped)
    else:
        mapped = map(lambda(x): fill if x is _EXCEPTION_FOUND else x, data)
    return mapped


def iexception_safe_map(func, data, exception=Exception, fill=ARGUMENT_NOT_SPECIFIED):
    """ the iterator version of exception_safe_map.

    NOTE: this may result in changed indices and length of input data

    A map function that filters out elements where the mapping function generated an exception.

    Args:
        func (lambda): the mapping function
        data (list): some iterable to map over
        exception (BaseException, optional): the specific type of exception to make safe, all other exceptions will be
            raised
        fill (object, optional): object to be used in place of the element that generated an exception. If not specified
            then the element will simply be omitted. Note that indices and length may change as a result.

    Returns:
        (list): iterator mapped with func and with all exception-generating elements removed
    """
    def safe_func(inp):
        try:
            return func(inp)
        except exception:
            return _EXCEPTION_FOUND

    mapped = imap(safe_func, data)
    if fill is ARGUMENT_NOT_SPECIFIED:
        mapped = ifilterfalse(lambda(x): x is _EXCEPTION_FOUND, mapped)
    else:
        mapped = imap(lambda(x): fill if x is _EXCEPTION_FOUND else x, data)
    return mapped