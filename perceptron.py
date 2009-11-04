# Copyright (c) 2009 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''Basic Perceptron library.

Perceptrons are supervised, discriminative models for classifying data points
(also called "events") into one of a number of discrete categories (also called
"outcomes"). The basic Perceptron is simply a vector W in event space that is
normal to a linear separating surface going through the origin. Data points that
fall on one side of the surface are in outcome (+), and data points on the other
side are in outcome (-). To determine the side of the surface on which some
event X falls, we compute the dot product of X and W and return the sign of the
resulting scalar. Learning takes place in the Perceptron whenever it makes an
incorrect classification from labeled data : X is simply added to W to minimally
correct the output of the Perceptron for this data point. This might invalidate
classifications for other events in the data set, but it is simple.

This particular library generalizes the Perceptron to K > 2 outcomes by
maintaining a surface for each of the K possible outcomes. Each surface can be
thought of as separating events in one outcome from events in all other
outcomes. At classification time, we compute the dot product of X with each of
the Ws. The outcome with the greatest resulting scalar value is returned as the
output.

This library also incorporates the "kernel trick" to generalize the notion of
the dot product to (almost) arbitrary high-dimensional spaces. Instead of
computing the dot product between X and W, we use a Kernel function that accepts
X and W as inputs and returns a scalar value indicating their similarity in some
(often much higher-dimensional) mapped comparison space.

Run a test on the library by calling this script from the command line.
'''

import numpy
from numpy import random as rng


class Kernel(object):
    '''A Kernel function maps two vectors onto a scalar.

    Kernels must be decomposable into the product of some function mapped to
    each of the input vectors.
    '''

    def __call__(self, a, b):
        '''Obtain a scalar from two vectors.'''
        raise NotImplemented


class DotProductKernel(Kernel):
    '''This Kernel calculates the standard dot product of two vectors.'''

    def __call__(self, a, b):
        return numpy.dot(a, b)


class RadialBasisKernel(Kernel):
    '''This Kernel returns the gaussian distance between the input vectors.'''

    def __init__(self, gamma):
        self._gamma = gamma

    def __call__(self, a, b):
        delta = b - a
        return numpy.exp(-self._gamma * (delta * delta).sum())


class PolynomialKernel(Kernel):
    '''This Kernel returns the dot product raised to some power.'''

    def __init__(self, degree, alpha=1.0):
        self._degree = degree
        self._alpha = alpha

    def __call__(self, a, b):
        return (numpy.dot(a, b) + self._alpha) ** self._degree


def _max_outcome(weights, kernel, event):
    '''Return the maximum scoring outcome for a set of weights.'''
    max_outcome = -1
    max_score = -numpy.inf
    for outcome in xrange(weights.shape[0]):
        score = kernel(weights[outcome, :], event)
        if score > max_score:
            max_outcome = outcome
            max_score = score
    return max_outcome, max_score


class Perceptron(object):
    '''A Perceptron is a discriminative machine learning algorithm.'''

    def __init__(self, event_size, outcome_size, kernel=None):
        '''Initialize this Perceptron.

        event_size: An integer indicating the dimensionality of the event space.
        outcome_size: An integer indicating the number of classes in the data.
        kernel: A Kernel object that we can use to compute the distance between
          a data point and our weight vector.
        '''
        assert outcome_size >= 2
        assert event_size >= 1
        self._weights = numpy.zeros((outcome_size, event_size), dtype='d')
        self._kernel = kernel or DotProductKernel()

    def _update(self, outcome, target):
        '''Update the weights for a single outcome toward a particular target.'''
        W = self._weights[outcome, :]
        W += target
        W /= numpy.sqrt((W * W).sum())

    def classify(self, event):
        '''Classify an event into one of our possible outcomes.

        event: A numpy vector of the dimensionality of our input space.

        Returns the most likely outcome, an integer in [0, outcome_size).
        '''
        return _max_outcome(self._weights, self._kernel, event)

    def learn(self, event, outcome):
        '''Adjust the hyperplane based on a classification attempt.

        event: A numpy vector of the dimensionality of our input space.
        outcome: An integer in [0, outcome_size) indicating the correct outcome
          for this event.
        '''
        prediction, _ = self.classify(event)
        if prediction is not outcome:
            self._update(prediction, -event)
            self._update(outcome, event)


class AveragedPerceptron(Perceptron):
    '''A weighted sum of individual Perceptrons.

    This Perceptron algorithm performs similarly to the basic Perceptron when
    learning from labeled data : Whenever the predicted outcome for an event
    differs from the true outcome, the weights of the Perceptron are updated to
    classify this new event correctly.

    However, in addition to updating the weights of the Perceptron in response
    to errors during learning, the AveragedPerceptron also makes a copy of the
    old weight matrix and adds it to a running sum of all past weight matrices.
    The sums are weighted by the number of iterations that each constituent
    weight matrix survived before making an error.

    At classification time, the AveragedPerceptron uses both the current weight
    matrix and the weighted sum of past matrices to make its decision.

    This is equivalent to the "voted perceptron" algorithm described by Freund
    and Schapire (1999). The averaging approach improves on the basic Perceptron
    algorithm by providing a "large margin" approach to handling datasets that
    are not linearly separable.
    '''

    def __init__(self, event_size, outcome_size, kernel=None):
        parent = super(AveragedPerceptron, self)
        parent.__init__(event_size, outcome_size, kernel)
        self._iterations = 0
        self._survived = 0
        self._history = self._weights.copy()

    def learn(self, event, outcome):
        self._iterations += 1
        self._survived += 1
        prediction, _ = _max_outcome(self._weights, self._kernel, event)
        if outcome is not prediction:
            self._history += self._weights * self._survived
            self._survived = 0
            self._update(prediction, -event)
            self._update(outcome, event)

    def classify(self, event):
        weights = self._history + self._weights * self._survived
        return _max_outcome(weights / self._iterations, self._kernel, event)


if __name__ == '__main__':
    centers = (
        (0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1),
        (0, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1),
        (0, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1),
        (0, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1),
        (0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        )

    def sample(variance):
        outcome = int(rng.random() * len(centers))
        event = rng.normal(centers[outcome], variance)
        event[0] = 1  # always set the bias to 1.
        return event / numpy.sqrt((event * event).sum()), outcome

    kwargs = dict(event_size=len(centers[0]), outcome_size=len(centers))

    learners = []
    for Klass in (Perceptron, AveragedPerceptron):
        learners.append(Klass(**kwargs))
        for d in (1, 2, 5):
            learners.append(Klass(kernel=PolynomialKernel(d), **kwargs))
        learners.append(Klass(kernel=RadialBasisKernel(0.5), **kwargs))

    print 'training data: 500 points drawn randomly from %d gaussians' % len(centers)
    print '% correct on 100 points drawn from the same gaussians'
    print 'Var\t| P\tP:p1\tP:p2\tP:p5\tP:r\t| A\tA:p1\tA:p2\tA:p5\tA:r'

    for variance in (0.1, 0.2, 0.5, 1, 2, 5, 10):
        # train all classifiers on a set of sample data.
        for _ in xrange(500):
            event, outcome = sample(variance)
            for p in learners:
                p.learn(event, outcome)

        # test on a different set of data.
        correct = [0] * len(learners)
        for _ in xrange(100):
            event, outcome = sample(variance)
            for i, p in enumerate(learners):
                prediction, _ = p.classify(event)
                if prediction is outcome:
                    correct[i] += 1

        # display some results.
        print '%4.1f' % variance, '\t|',
        for i, p in enumerate(learners):
            print '%3d' % correct[i], '\t',
            if i == 4:
                print '|',
        print
