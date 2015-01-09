import numpy as np
import theano.tensor as T
import theano


def make_shared(d, to_int=False, **kwargs):
    sd = theano.shared(np.asarray(d, dtype=theano.config.floatX),
                       **kwargs)
    if to_int:
        return T.cast(sd, 'int32')
    return sd


def make_probability(init, shape=None, **kwargs):
    if shape:
        init = np.ones(shape) * init
    logit_p = np.log(init / (1 - init))
    logit_p = make_shared(logit_p, **kwargs)
    return 1 / (1 + T.exp(-logit_p)), logit_p
