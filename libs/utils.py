import datetime
from random import randint, sample
import theano
import theano.tensor as T
import numpy

# I should probably split these into separate files but it would kind of be a
# waste of a files right now since they'll probably all be in separate ones


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


def random_unique_subset(v, percentage=.8):
    u_v = numpy.unique(v)
    s_v = sample(u_v, int((1 - percentage) * len(u_v)))
    return sum((v == s for s in s_v))


def combine_dict(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out

if __name__ == '__main__':
    print random_unique_subset(numpy.asarray([1, 2, 3, 3, 2, 1]), percentage=.6)
