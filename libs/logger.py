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


def log_args(currentframe, include_kwargs=False):
    _, _, _, arg_dict = inspect.getargvalues(currentframe)
    explicit_args = [(k, v) for k, v in arg_dict.iteritems()
                     if isinstance(v, (int, long, float, str))]
    keyword_args = arg_dict.get('kwargs', {}).items() if include_kwargs else []
    arg_summary = ', '.join(['{0}={1}'.format(*v) for v in
                             explicit_args + keyword_args])
    log(arg_summary)


def gen_log_name(uid=None):
    if uid is None:
        uid = str(randint(0, 99999))
    return '{time}_{uid}.log'.format(
        time=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        uid=uid)
