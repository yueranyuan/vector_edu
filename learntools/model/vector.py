import numpy
import theano
import theano.tensor as T
from learntools.model.mlp import MLP


class VectorLayer(object):
    def __init__(self, rng, indices, full_input, vectors=None, n_skills=4600,
                 vector_length=30, mutable=True):
        self.indices = indices
        self.mutable = mutable

        if vectors is None:
            vectors = numpy.asarray(rng.uniform(low=0, high=1, size=(n_skills, vector_length)),
                                    dtype=theano.config.floatX)
        else:
            vectors = vectors.astype(dtype=theano.config.floatX)
        self.skills = theano.shared(vectors, borrow=True)
        skill_i = T.cast(full_input[self.indices], 'int32')
        self.output = self.skills[skill_i[:, 0]]
        self.params = [self.skills] if mutable else []


class VMLP(object):
    def __init__(self, rng, input, vector_length, n_skills, n_hidden, n_out, full_input):
        self.vectors = VectorLayer(rng=rng,
                                   input=input,
                                   full_input=full_input,
                                   n_skills=n_skills,
                                   vector_length=vector_length)

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
        self.dropout = self.MLP.dropout
