'''Basic perceptron library.

A Perceptron is a basic machine learning algorithm with a long and colored
past. Perceptrons are supervised, discriminative models requiring a set of
labeled data.
'''

import numpy
import collections


class Kernel(object):
    '''A Kernel function maps two vectors onto a scalar. Kernels must be
    decomposable into the dot product of two vectors of possibly infinite
    dimension.'''

    def __call__(self, a, b):
        '''Obtain a scalar from two vectors.'''
        raise NotImplemented

    def reset(self):
        '''Reset this Kernel object. Not needed for most kernels.'''
        pass


class DotProductKernel(Kernel):
    '''This Kernel calculates the standard dot product of two vectors.'''

    def __call__(self, a, b):
        return numpy.dot(a, b)


class RadialBasisKernel(Kernel):
    '''This Kernel '''

    def __init__(self, gamma, decay=1.0):
        self._gamma_0 = self._gamma = gamma
        self._decay = decay

    def __call__(self, a, b):
        d = b - a
        self._gamma *= self._decay
        return numpy.exp(self._gamma * numpy.dot(d, d))

    def reset(self):
        self._gamma = self._gamma_0


class PolynomialKernel(Kernel):
    def __init__(self, degree, alpha=1.0):
        self._degree = degree
        self._alpha = alpha

    def __call__(self, a, b):
        return (numpy.dot(a, b) + self._alpha) ** self._degree


class Perceptron(object):
    def __init__(self, dimension=None, weights=None, kernel=None):
        self._weights = weights or numpy.ones((dimension, ), dtype='d')
        self._kernel = kernel or DotProductKernel()

    def weights(self):
        return self._weights

    def kernel(self):
        return self._kernel

    def classify(self, x, label=None):
        result = self._kernel(self._weights, x)
        if label is not None and label * result < 0:
            self._weights += x
        return result


class VotedPerceptron(collections.deque):
    def __init__(self,
                 dimension=None,
                 weights=None,
                 kernel=None,
                 max_voters=0):
        self._voter = Perceptron(dimension=dimension,
                                 weights=weights,
                                 kernel=kernel)
        self._weight = 1
        self._max_voters = max_voters

    def classify(self, x, label=None):
        aggregate = 0.0
        for weight, voter in self:
            aggregate += weight * voter.classify(x)

        result = self._voter.classify(x, label=label)
        aggregate += self._weight * result

        if label is not None:
            if label * result < 0:
                self.append((self._weight, self._voter))
                if len(self) > self._max_voters > 0:
                    self.popleft()
                self._voter = Perceptron(weights=self._voter.weights(),
                                         kernel=self._voter.kernel())
                self._weight = 1
            else:
                self._weight += 1

        return aggregate


if __name__ == '__main__':
    p = Perceptron(3)
    for point, label in (((1, 1, 1), 1),
                         ((-1, -1, -1), -1),
                         ):
        correct = p.classify(point, label) * label > 0
