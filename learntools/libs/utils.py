import operator
from itertools import chain, imap, ifilterfalse

import numpy
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


def normalize_table(table):
    table = numpy.array(table)
    mins = table.min(axis=0)
    maxs = table.max(axis=0)
    norm_table = (table - mins) / (maxs - mins)
    if numpy.any(numpy.isnan(norm_table)):
        # TODO: issue warning all nan
        print "Warning: normalized table contains nans"
        if not numpy.any(numpy.isnan(table)):
            print "Warning: nans were not present in input table"
    return norm_table


# converts an index array into the corresponding mask
# example: [1, 3, 4] -> [False, True, False, True, True]
def idx_to_mask(idxs, mask_len=None):
    if not mask_len:
        mask_len = max(idxs) + 1
    mask = numpy.array([False] * mask_len)
    mask[idxs] = True
    return mask


def mask_to_idx(mask):
    return numpy.nonzero(mask)[0]


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