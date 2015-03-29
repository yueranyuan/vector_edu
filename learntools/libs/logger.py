import datetime
from random import randint
import json


__LOG_FILE__ = None


def set_log_file(log_file):
    global __LOG_FILE__
    __LOG_FILE__ = log_file


def get_log_file():
    return __LOG_FILE__


def log(txt, also_print=False):
    if also_print:
        print txt
    if __LOG_FILE__ is None:
        raise Exception('log module must first be initialized with libs.logger.set_log_file')
    with open(__LOG_FILE__, 'a+') as f:
        f.write('{0}\n'.format(txt))


def log_me(start_with=None):
    def logging_decorator(func):
        echo_wrapper = echo(func, write=log, format=format_all_args_json)

        def func_wrapper(*args, **kwargs):
            if start_with:
                log(start_with, True)
            return echo_wrapper(*args, **kwargs)
        return func_wrapper
    return logging_decorator


def gen_log_name(uid=None):
    if uid is None:
        uid = str(randint(0, 99999))
    return '{time}_{uid}.log'.format(
        time=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        uid=uid)


# the following two functions are copied from http://wordaligned.org/articles/echo
# and lightly modified

import sys


def format_arg_value(arg_val):
    """ Return a string representing a (name, value) pair.

    Examples:
        >>> format_arg_value(('x', (1, 2, 3)))
        'x=(1, 2, 3)'
    """
    arg, val = arg_val
    return "%s=%r" % (arg, val)


def format_all_args_json(func_name=None, named=None, nameless=None):
    """ format arguments as json.

    Format arguments as Json. Ignores unnamed arguments. Ignores objects.

    Examples:
        >>> format_all_args('f1', named=[('x', 1), ('y', 2)], nameless=[3, 'z'])
        Arguments to f1: {x: 1, y: 2}
    """

    def _is_serializable(v):
        try:
            json.dumps(v)
            return True
        except TypeError:
            return False
    args_filtered = filter(lambda(k, v): isinstance(k, str) and _is_serializable(v), named)
    arg_dict = dict(args_filtered)
    return 'Arguments to {func_name}: {dump}'.format(func_name=func_name, dump=json.dumps(arg_dict))


def format_all_args(func_name=None, named=None, nameless=None):
    """ format arguments as comma delimited "name=value" pairs

    Examples:
        >>> format_all_args('f1', named=[('x', 1), ('y', 2)], nameless=[3, 'z'])
        'x=1, y=2, 3, z'
    """
    if named is None:
        named = []
    if nameless is None:
        nameless = []
    named_formatted = map(format_arg_value, named)
    nameless_formatted = map(repr, nameless)
    args = named_formatted + nameless_formatted
    return ", ".join(args)


def echo(fn, write=sys.stdout.write, format=format_all_args):
    """ Echo calls to a function.

    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools
    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    func_name = fn.__name__
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))

    @functools.wraps(fn)
    def wrapped(*v, **k):
        # Collect function arguments by chaining together positional,
        # defaulted, extra positional and keyword arguments.
        positional = zip(argnames, v)
        defaulted = [(a, argdefs[a])
                     for a in argnames[len(v):] if a not in k]
        keyword = k.items()
        nameless = v[argcount:]
        named = positional + defaulted + keyword
        args_formatted = format(func_name=func_name, named=named, nameless=nameless)
        write(args_formatted)
        return fn(*v, **k)
    return wrapped
