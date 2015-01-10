import datetime
from random import randint
import inspect


__LOG_FILE__ = None


def set_log_file(log_file):
    global __LOG_FILE__
    __LOG_FILE__ = log_file


def log(txt, also_print=False):
    if also_print:
        print txt
    if __LOG_FILE__ is None:
        raise Exception('log module must first be initialized with libs.logger.set_log_file')
    with open(__LOG_FILE__, 'a+') as f:
        f.write('{0}\n'.format(txt))


def log_me(start_with=None):
    def logging_decorator(func):
        echo_wrapper = echo(func, write=log)

        def func_wrapper(*args, **kwargs):
            if start_with:
                log(start_with, True)
            return echo_wrapper(*args, **kwargs)
        return func_wrapper
    return logging_decorator


def log_args(currentframe, include_kwargs=False, exclude=None):
    _, _, _, arg_dict = inspect.getargvalues(currentframe)
    explicit_args = [(k, v) for k, v in arg_dict.iteritems()
                     if isinstance(v, (int, long, float, str))]
    keyword_args = arg_dict.get('kwargs', {}).items() if include_kwargs else []
    print explicit_args
    all_args = explicit_args + keyword_args
    all_args = filter(lambda (k, v): k not in exclude, all_args)

    log(', '.join(['{0}={1}'.format(*v) for v in all_args]))


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

    >>> format_arg_value(('x', (1, 2, 3)))
    'x=(1, 2, 3)'
    """
    arg, val = arg_val
    return "%s=%r" % (arg, val)


def echo(fn, write=sys.stdout.write):
    """ Echo calls to a function.

    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools
    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))

    @functools.wraps(fn)
    def wrapped(*v, **k):
        # Collect function arguments by chaining together positional,
        # defaulted, extra positional and keyword arguments.
        positional = map(format_arg_value, zip(argnames, v))
        defaulted = [format_arg_value((a, argdefs[a]))
                     for a in argnames[len(v):] if a not in k]
        nameless = map(repr, v[argcount:])
        keyword = map(format_arg_value, k.items())
        args = positional + defaulted + nameless + keyword
        log(", ".join(args))
        return fn(*v, **k)
    return wrapped
