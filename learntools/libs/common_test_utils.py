'''utilities to be used by tests.

awkward name module name? yeah, I didn't want this to be confused with
test_utils.py which would be a test of utils'''
import random
import os

from learntools.libs.logger import set_log_file


class Log(object):
    def __init__(self, log_name):
        self.log_name = log_name

    def __enter__(self):
        set_log_file(self.log_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.remove(self.log_name)
        except WindowsError:
            if os.path.exists(self.log_name):
                raise
        return False


# TODO: convert try-finally to context
def use_logger_in_test(func):
    import functools

    @functools.wraps(func)
    def decorated_test(*args, **kwargs):
        with Log("temp_testlog_{}.log".format(random.randint(0, 99999))):
            return func(*args, **kwargs)
    return decorated_test
