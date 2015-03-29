import theano.tensor as T


def sigmoid(x):
    return 1 / (1 + T.exp(-x))


def rectifier(x):
    return T.maximum(x, 0)


def p_rectifier(p):
    def func(x):
        return T.maximum(x, T.minimum(0, p * x))
    return func


def neg_log_loss(p, y):
    return -T.sum(T.log(p.T)[T.arange(y.shape[0]), y])


def mean_neg_log_loss(p, y):
    return -T.mean(T.log(p.T)[T.arange(y.shape[0]), y])
