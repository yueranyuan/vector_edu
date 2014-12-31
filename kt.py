import math

import theano.tensor as T
from theano import function as Tfunction
import numpy as np

from libs.utils import make_shared


def neg_log_loss(p, y):
    return -T.mean(T.log(p.T)[T.arange(y.shape[0]), y])

p = make_shared([[.5, .5], [.4, .6], [.3, .7]])
y = make_shared([0, 1, 1], to_int=True)
loss = neg_log_loss(p, y)

learning_rate = 0.01
N = 3
p_L0 = 0.4
p_T = 0.5
p_G = 0.1
p_S = 0.1

tp_L = make_shared([p_L0] + np.zeros(N))
tp_L0 = make_shared(p_L0)


def make_probability(p, **kwargs):
    logit_p = math.log(p / (1 - p))
    logit_p = make_shared(logit_p, **kwargs)
    return 1 / (1 + T.exp(-logit_p)), logit_p

tp_T, t_T = make_probability(p_T, name='p(T)')
tp_G, t_G = make_probability(p_G, name='p(G)')
tp_S, t_S = make_probability(p_S, name='p(S)')
print tp_S.eval()

# TODO: do a base-case due to the broken initial state estimation system

i = T.ivector('i')
tp_L_view = tp_L[i - 1] + (1 - tp_L[i - 1]) * tp_T
tp_C = tp_L_view * (1 - tp_S) + (1 - tp_L_view) * tp_G
tp_y = T.stack(1 - tp_C, tp_C)
loss = neg_log_loss(tp_y, y[i])

params = [t_T, t_G, t_S]
update_L = [(tp_L, T.set_subtensor(tp_L[i], tp_L_view))]
update_parameters = [(param, param - learning_rate * T.grad(loss, param))
                     for param in params]
updates = update_parameters + update_L
train = Tfunction(
    updates=updates,
    inputs=[i],
    outputs=[loss, tp_C]
)
train_idxs = [1, 2]

for i in range(len(train_idxs)):
    print train(train_idxs[i:i + 1])

n_epochs = 100
for i in range(n_epochs):
    print train(train_idxs)
