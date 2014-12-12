import numpy
import theano
import theano.tensor as T
from mlp import MLP


class VectorLayer(object):
    def __init__(self, rng, input, full_input, n_skills=4600, vector_length=30):
        self.skills = theano.shared(numpy.asarray(rng.uniform(low=0, high=1,
                                                              size=(n_skills, vector_length)),
                                                  dtype=theano.config.floatX),
                                    borrow=True)
        skill_i = T.cast(input, 'int32')
        self.output = self.skills[skill_i[:, 0]]
        self.m = theano.shared(reindex(full_input.get_value(borrow=True),
                                       self.skills.get_value(borrow=True)),
                               borrow=True)

    def get_updates(self, cost, index, batch_size, learning_rate):
        gx = T.grad(cost, self.output)
        gskills = T.dot(self.m[:, index * batch_size: (index + 1) * batch_size], gx)
        return [(self.skills, self.skills - learning_rate * gskills)]


class VMLP(object):
    def __init__(self, rng, input, vector_length, n_skills, n_hidden, n_out, full_input):
        self.vectors = VectorLayer(rng=rng,
                                   input=input,
                                   full_input=full_input,
                                   n_skills=n_skills,
                                   vector_length=30)

        self.MLP = MLP(
            rng=rng,
            n_in=vector_length,
            input=self.vectors.output,
            n_hidden=n_hidden,
            n_out=n_out)

        self.L1 = self.MLP.L1
        self.L2_sqr = self.MLP.L2_sqr

        self.negative_log_likelihood = self.MLP.negative_log_likelihood
        self.errors = self.MLP.errors
        self.output = self.MLP.output

        self.params = self.MLP.params
        self.get_updates = self.vectors.get_updates


def reindex(skills, skillVecs, y=None):
    y = y or numpy.zeros([len(skillVecs), len(skills)])
    for si in range(len(skillVecs)):
        x = numpy.array(skills)
        i = numpy.where(x == si)[0]
        y[si][i] = 1
    return y
