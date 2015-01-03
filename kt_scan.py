from itertools import groupby

import theano
import theano.tensor as T
import numpy as np

from libs.utils import make_shared


def make_probability(init, shape=None, **kwargs):
    if shape:
        init = np.ones(shape) * init
    logit_p = np.log(init / (1 - init))
    logit_p = make_shared(logit_p, **kwargs)
    return 1 / (1 + T.exp(-logit_p)), logit_p


def neg_log_loss(p, y):
    return -T.mean(T.log(p.T)[T.arange(y.shape[0]), y])


def step(correct_i, prev_L, prev_p_C, P_T, P_S, P_G):
    Ln = prev_L + (1 - prev_L) * P_T
    p_C = prev_L * (1 - P_S) + (1 - prev_L) * P_G
    return Ln, p_C

# prepare data
correct_y = np.asarray([0, 1, 1, 0, 0, 1, 0, 1, 1])
skill_x = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0])
subject_x = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1])
n_skills = np.max(skill_x) + 1
n_subjects = np.max(subject_x) + 1
N = len(correct_y)

# prepare parameters
p_T = 0.5
p_G = 0.1
p_S = 0.2
p_L0 = 0.2
parameter_base = np.ones((n_skills, n_subjects))
tp_L0, t_L0 = make_probability(parameter_base * p_L0, name='L0')
tp_T, t_T = make_probability(parameter_base * p_T, name='p(T)')
tp_G, t_G = make_probability(parameter_base * p_G, name='p(G)')
tp_S, t_S = make_probability(parameter_base * p_S, name='p(S)')

# declare and prepare variables for theano
i = T.ivector('i')
dummy_float = make_shared(0, name='dummy')
skill_i, subject_i = T.iscalars('skill_i', 'subject_i')
correct_y = make_shared(correct_y, to_int=True)

# set up theano functions
((results, p_C), updates) = theano.scan(fn=step,
                                        sequences=correct_y[i],
                                        outputs_info=[tp_L0[skill_i, subject_i],
                                                      dummy_float],
                                        non_sequences=[tp_T[skill_i, subject_i],
                                                       tp_G[skill_i, subject_i],
                                                       tp_S[skill_i, subject_i]])

p_y = T.stack(1 - p_C, p_C)
loss = neg_log_loss(p_y, correct_y[i])

learning_rate = 0.02
params = [t_T, t_G, t_S, t_L0]
update_parameters = [(param, param - learning_rate * T.grad(loss, param))
                     for param in params]
f = theano.function(inputs=[i, skill_i, subject_i],
                    updates=update_parameters,
                    outputs=[loss, results])

# train this thing
n_epochs = 54
for epoch in range(n_epochs):
    print 'epoch'
    for subject, subject_idxs in groupby(xrange(N), key=lambda i: subject_x[i]):
        for skill, skill_idxs in groupby(subject_idxs, key=lambda i: skill_x[i]):
            print f(list(skill_idxs), skill, subject)
