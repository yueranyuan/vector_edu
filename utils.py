import datetime
from random import randint
import theano
import theano.tensor as T
import numpy


def gen_log_name(uid=None):
    if uid is None:
        uid = str(randint(0, 99999))
    return '{time}_{uid}.log'.format(
        time=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        uid=uid)


def make_shared(d, to_int=False):
    sd = theano.shared(numpy.asarray(d, dtype=theano.config.floatX), borrow=True)
    if to_int:
        return T.cast(sd, 'int32')
    return sd
