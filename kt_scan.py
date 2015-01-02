import theano
import theano.tensor as T

from libs.utils import make_shared


def neg_log_loss(p, y):
    return -T.mean(T.log(p.T)[T.arange(y.shape[0]), y])


def step(correct_i, prev_L, prev_p_C, P_T, P_S, P_G):
    Ln = prev_L + (1 - prev_L) * P_T
    p_C = prev_L * (1 - P_S) + (1 - prev_L) * P_G
    return Ln, p_C


L0 = T.scalar('L0')
P_T = make_shared(0.5)
P_G = make_shared(0.1)
P_S = make_shared(0.2)
correct_y = make_shared([0, 0, 1, 1], to_int=True)

((results, p_C), updates) = theano.scan(fn=step,
                               sequences=correct_y,
                               outputs_info=[L0, T.ones_like(L0)],
                               non_sequences=[P_T, P_G, P_S])

p_y = T.stack(1 - p_C, p_C)
loss = neg_log_loss(p_y, correct_y)

f = theano.function(inputs=[L0], outputs=[loss, results, p_C, p_y])

print f(0.2)
