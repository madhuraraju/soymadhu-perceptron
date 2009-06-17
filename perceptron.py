'''Basic perceptron library.

A Perceptron is a basic machine learning algorithm with a long and colored
past. Perceptrons are supervised, discriminative models requiring a set of
labeled data.
'''

import numpy
import collections


def norm(x):
    return numpy.dot(x, x)


class Kernel(object):
    '''A Kernel function maps two vectors onto a scalar.

    Kernels must be decomposable into the product of some function mapped to
    each of the input vectors.
    '''

    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __repr__(self):
        kwargs = ', '.join('%s=%r' % i for i in self.__kwargs.iteritems())
        return '%s(%s)' % (self.__class__.__name__, kwargs)

    def __call__(self, a, b):
        '''Obtain a scalar from two vectors.'''
        raise NotImplemented

    def copy(self):
        '''Return a copy of this kernel object.'''
        return self.__class__(**self.__kwargs)

    def reset(self):
        '''Reset this Kernel object. Not needed for most kernels.'''
        pass


class DotProductKernel(Kernel):
    '''This Kernel calculates the standard dot product of two vectors.'''

    def __call__(self, a, b):
        return numpy.dot(a, b)


class RadialBasisKernel(Kernel):
    '''This Kernel returns the gaussian distance between the input vectors.'''

    def __init__(self, gamma, decay=1.0):
        super(RadialBasisKernel, self).__init__(gamma=gamma, decay=decay)
        self._gamma_0 = self._gamma = gamma
        self._decay = decay

    def __call__(self, a, b):
        self._gamma *= self._decay
        delta = b - a
        return numpy.exp(self._gamma * norm(b - a))


class PolynomialKernel(Kernel):
    '''This Kernel returns the dot product raised to some power.'''

    def __init__(self, degree, alpha=1.0):
        super(PolynomialKernel, self).__init__(degree=degree, alpha=alpha)
        self._degree = degree
        self._alpha = alpha

    def __call__(self, a, b):
        return (numpy.dot(a, b) + self._alpha) ** self._degree


class Perceptron(object):
    '''A perceptron is a simple discriminative machine learning algorithm.'''

    def __init__(self, dimension=None, weights=None, kernel=None):
        if weights is None:
            self._weights = numpy.random.normal(size=(dimension, ))
        else:
            self._weights = numpy.array(weights)

        if kernel is None:
            self._kernel = DotProductKernel()
        else:
            self._kernel = kernel

    def __repr__(self):
        return 'Perceptron(weights=%r, kernel=%r)' % (
            tuple(self._weights), self._kernel)

    def weights(self):
        return self._weights

    def kernel(self):
        return self._kernel

    def classify(self, x):
        return self._kernel(self._weights, x)

    def learn(self, x, label):
        if label ^ (self.classify(x) > 0):
            self._weights += x
            self._weights /= norm(self._weights)


class VotedPerceptron(object):
    '''A voted perceptron is a weighted sum of individual perceptrons.'''

    def __init__(self,
                 dimension=None,
                 weights=None,
                 kernel=None,
                 max_voters=0):
        self._voters = []
        self._max_voters = max_voters
        self._voter = Perceptron(dimension=dimension,
                                 weights=weights,
                                 kernel=kernel)
        self._weight = 1

    def voters(self):
        return self._voters

    def classify(self, x):
        '''Classify an unlabeled data point.'''
        aggregate = 0.0
        for weight, voter in self._voters:
            aggregate += weight * voter.classify(x)
        return aggregate + self._weight * self._voter.classify(x)

    def learn(self, x, label):
        '''Learn from a labeled data point.'''
        if label ^ (self._voter.classify(x) > 0):
            # the current perceptron got this point wrong. we add this
            # perceptron to our list of voters, along with the current weight.
            self._voters.append((self._weight, self._voter))

            # we might have to discard a voter. if so, discard the one with the
            # lowest weight.
            if len(self._voters) > self._max_voters > 0:
                self._voters.sort(reverse=True)
                self._voters.pop()

            # create a new perceptron with the old one's weights and kernel
            # function.
            weights = self._voter.weights() + x
            weights /= norm(weights)

            kernel = self._voter.kernel().copy()

            self._voter = Perceptron(weights=weights, kernel=kernel)
            self._weight = 0

        # increase the weight for this voter by 1. if we didn't replace the
        # current voter, this effectively increases its voting power.
        self._weight += 1


if __name__ == '__main__':
    import random

    M = 1001
    N = 101
    centers = [(1, 0, 0), (-1, 0, 0)]

    def sample():
        klass = random.randint(0, 1)
        point = numpy.random.normal(centers[klass], 1)
        return point, bool(klass)

    kwargs = dict(dimension=len(centers[0]))
    p = Perceptron(**kwargs)

    kwargs['max_voters'] = 5
    v = VotedPerceptron(**kwargs)

    kwargs['kernel'] = PolynomialKernel(2)
    k = VotedPerceptron(**kwargs)

    for _ in xrange(M):
        point, label = sample()
        p.learn(point, label)
        v.learn(point, label)
        k.learn(point, label)

    p_correct = 0
    v_correct = 0
    k_correct = 0
    for _ in xrange(N):
        point, label = sample()
        if not ((p.classify(point) > 0) ^ label):
            p_correct += 1
        if not ((v.classify(point) > 0) ^ label):
            v_correct += 1
        if not ((k.classify(point) > 0) ^ label):
            k_correct += 1

    print '%d training / %d test samples' % (M, N)
    print 'perceptron - %.1f %% correct' % (100.0 * p_correct / N)
    print '  0:', repr(p)
    print 'voting - %.1f %% correct' % (100.0 * v_correct / N)
    for w, p in v.voters(): print '  %d: %r' % (w, p)
    print 'kernel - %.1f %% correct' % (100.0 * k_correct / N)
    for w, p in k.voters(): print '  %d: %r' % (w, p)

