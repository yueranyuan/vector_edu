import random
import os

from learntools.kt.kt import build_model
from learntools.kt.data import prepare_fake_data
from learntools.model.train import train_model
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


@use_logger_in_test
def smoke_build_model(build_model):
    set_log_file("testlog.log")
    prepared_data = prepare_fake_data()

    f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval = (
        build_model(prepared_data, batch_size=1))

    best_validation_loss, best_epoch = (
        train_model(f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval,
                    n_epochs=100))

    assert best_validation_loss > 0.5
    assert best_epoch > 0


def test_kt_smoke():
    smoke_build_model(build_model)
