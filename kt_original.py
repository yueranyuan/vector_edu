import theano.tensor as T
from theano import function as Tfunction
import numpy as np

from libs.utils import make_shared


def neg_log_loss(p, y):
    return -T.mean(T.log(p.T)[T.arange(y.shape[0]), y])

correct_y = np.asarray([0, 1, 1, 0, 0, 1])
skill_x = np.asarray([0, 0, 0, 1, 1, 1])
n_skills = np.max(skill_x) + 1
N = len(correct_y)

learning_rate = 0.02
skill_starts = np.concatenate(([0], np.nonzero(skill_x[:-1] - skill_x[1:])[0] + 1))
p_L = np.zeros(N)
p_T = 0.5
p_G = 0.1
p_S = 0.1
correct_y = make_shared(correct_y, to_int=True)
skill_x = make_shared(skill_x, to_int=True)


def make_probability(init, **kwargs):
    logit_p = np.log(init / (1 - init))
    logit_p = make_shared(logit_p, **kwargs)
    return 1 / (1 + T.exp(-logit_p)), logit_p

tp_Ln = make_shared(p_L)
p_L0 = np.ones(len(skill_starts)) * 0.4
tp_L0, t_L0 = make_probability(p_L0, name='p_L0')
tp_L = T.concatenate((tp_L0[0:1], tp_Ln[1:3], tp_L0[1:], tp_Ln[4:]))

tp_T, t_T = make_probability(np.ones(n_skills) * p_T, name='p(T)')
tp_G, t_G = make_probability(np.ones(n_skills) * p_G, name='p(G)')
tp_S, t_S = make_probability(np.ones(n_skills) * p_S, name='p(S)')

# TODO: do a base-case due to the broken initial state estimation system

i = T.ivector('i')
tp_L_view = tp_L[i - 1] + (1 - tp_L[i - 1]) * tp_T[skill_x[i - 1]]
tp_C = tp_L_view * (1 - tp_S[skill_x[i]]) + (1 - tp_L_view) * tp_G[skill_x[i]]
tp_y = T.stack(1 - tp_C, tp_C)
loss = neg_log_loss(tp_y, correct_y[i])

params = [t_T, t_G, t_S, t_L0]
update_L = [(tp_Ln, T.set_subtensor(tp_Ln[i], tp_L_view))]
update_parameters = [(param, param - learning_rate * T.grad(loss, param))
                     for param in params]
updates = update_parameters + update_L
train = Tfunction(
    updates=updates,
    inputs=[i],
    outputs=[tp_C, tp_T, tp_L]
)
train_idxs = [1, 2, 4, 5]

for i in range(len(train_idxs)):
    print train(train_idxs[i:i + 1])

n_epochs = 54
for i in range(n_epochs):
    print train(train_idxs)
