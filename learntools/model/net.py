import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from learntools.model.math import rectifier


class NetworkComponent(object):
    '''Abstract network object that is not meant to be used. Holds some convenience functions
    and the signature for NetworkComponents'''
    def __init__(self, name):
        '''create all theano variables at initialization

        Args:
            name (str): the name of the network. Use subname() to generate names for
                all theano variables which are a part of this network so that debug will
                display an appropriate name for the variable that indicates its place in the
                network hierarchy'''
        self.name = name

    def instance(self, x, **kwargs):
        '''generate the theano variable for the output of this network given the input x

        Args:
            x: a theano variable that represents the input to this network

        Returns:
            a theano variable that represents the output of this network
        '''
        raise Exception('instance has not been implemented for this network')

    def subname(self, suffix):
        '''generate a name for a theano variable. Make sure all theano variables that are a
        part of this network are named using names generated by this function

        Args:
            suffix (str): the desired name of the theano variable

        Returns:
            (str): the full name of a theano variable with the name of this network appended'''
        return '{root}_{suffix}'.format(root=self.name, suffix=suffix)

    @property
    def L1(self):
        '''L1 regularization value'''
        if hasattr(self, '_L1'):
            return self._L1
        self._L1 = sum([c.L1 for c in self.components])
        self._L1.name = self.subname('L1')
        return self._L1

    @L1.setter
    def L1(self, L1):
        self._L1 = L1
        self._L1.name = self.subname('L1')

    @property
    def L2_sqr(self):
        '''L2 regularization value'''
        if hasattr(self, '_L2_sqr'):
            return self._L2_sqr
        self.L2_sqr = sum([c.L2_sqr for c in self.components])
        return self.L2_sqr

    @L2_sqr.setter
    def L2_sqr(self, L2_sqr):
        self._L2_sqr = L2_sqr
        self._L2_sqr.name = self.subname('L2_sqr')

    @property
    def params(self):
        '''all differentiable theano variables of this network'''
        if hasattr(self, '_params'):
            return self._params
        self.params = sum([c.params for c in self.components], [])
        return self.params

    @params.setter
    def params(self, params):
        self._params = params


# inspired by https://github.com/mdenil/dropout/blob/master/mlp.py
class HiddenLayer(NetworkComponent):
    def __init__(self, rng, n_in, n_out=None, W=None, b=None,
                 activation=rectifier, dropout=None, name='hiddenlayer'):
        super(HiddenLayer, self).__init__(name=name)
        self.dropout = T.scalar('dropout') if dropout is None else dropout
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=self.subname('W'), borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=self.subname('b'), borrow=True)

        self.W = W
        self.b = b

        self.activation = activation
        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()

    def instance(self, x, **kwargs):
        # dropouts
        mask = self.srng.binomial(n=1, p=1 - self.dropout, size=x.shape)
        # cast because int * float32 = float64 which does not run on GPU
        x = x * T.cast(mask, theano.config.floatX)
        lin_output = (T.dot(x, self.W) + self.b) * (1 / (1 - self.dropout))
        return self.activation(lin_output)


class HiddenNetwork(NetworkComponent):
    def __init__(self, n_in, size, input=None, name='hiddennetwork', **kwargs):
        super(HiddenNetwork, self).__init__(name=name)
        self.name = name
        self.layers = []
        for i, (n_in_, n_out_) in enumerate(zip([n_in] + size, size)):
            self.layers.append(HiddenLayer(n_in=n_in_,
                                           n_out=n_out_,
                                           name=self.subname('layer{i}'.format(i=i)),
                                           **kwargs))
        self.n_out = n_out_
        self.components = self.layers

    def instance(self, x, **kwargs):
        inp = x
        for layer in self.layers:
            inp = layer.instance(inp)
        return inp


class ConvolutionalLayer(NetworkComponent):
    def __init__(self, rng, n_in, n_out=None, W=None, b=None, field_width=3,
                 activation=rectifier, dropout=None, name='convolutionallayer'):
        super(ConvolutionalLayer, self).__init__(name=name)
        self.dropout = T.scalar('dropout') if dropout is None else dropout
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(1, 1, field_width, 1)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=self.subname('W'), borrow=True)

        if b is None:
            b_values = numpy.zeros((1,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=self.subname('b'), borrow=True)

        self.W = W
        self.b = b

        self.activation = activation
        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()

    def instance(self, x, **kwargs):
        # dropouts
        mask = self.srng.binomial(n=1, p=1 - self.dropout, size=x.shape)
        # cast because int * float32 = float64 which does not run on GPU
        x = x * T.cast(mask, theano.config.floatX)
        
        x_reshaped = T.reshape(x, (x.shape[0], 1, x.shape[1], 1), ndim=4)
        conv_output = conv.conv2d(x_reshaped, self.W)
        lin_output = (conv_output + self.b.dimshuffle('x', 0, 'x', 'x')) * (1 / (1 - self.dropout))
        ret = self.activation(lin_output.reshape((lin_output.shape[0], lin_output.shape[2])))
        return ret


class ConvolutionalNetwork(NetworkComponent):
    def __init__(self, n_in, size, input=None, name='convolutionalnetwork', **kwargs):
        super(ConvolutionalNetwork, self).__init__(name=name)
        self.name = name
        self.layers = []
        for i, (n_in_, n_out_) in enumerate(zip([n_in] + size, size)):
            self.layers.append(ConvolutionalLayer(n_in=n_in_,
                                           n_out=n_out_,
                                           name=self.subname('layer{i}'.format(i=i)),
                                           **kwargs))
        self.n_out = n_out_
        self.components = self.layers

    def instance(self, x, **kwargs):
        shape = x.shape
        inp = x
        for layer in self.layers:
            inp = layer.instance(inp)
        return inp
