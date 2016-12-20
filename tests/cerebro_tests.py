import numpy as np
from sklearn import datasets, metrics
from nose.tools import *
from cerebro.activations import ReLU, Softmax
from cerebro.core import Cerebro
from cerebro.initializers import Uniform
from cerebro.layers import Input, FeedForward
from cerebro.costs import CrossEntropy
from cerebro.optimizers import GradientDescent

from unittest import TestCase


class TestCerebro(TestCase):
    def setUp(self):
        # Load data
        self.X, self.t = datasets.make_circles(n_samples=100, shuffle=False, factor=0.3, noise=0.1)

        self.num_inputs = self.X.shape[1]
        self.num_outputs = 2

        # Create one-hot target
        self.T = np.zeros((self.X.shape[0], self.num_outputs))
        self.T[self.t == 1, 1] = 1
        self.T[self.t == 0, 0] = 1

    def tearDown(self):
        print "TEAR DOWN!"

    def test_basic(self):
        # Define a network
        net = Cerebro()

        net.add(Input(size=self.num_inputs))
        net.add(FeedForward(size=10, activation=ReLU(), initializer=Uniform))
        net.add(FeedForward(size=self.T.shape[1], activation=Softmax(), initializer=Uniform))

        # Choose loss function and Optimizer
        cost = CrossEntropy()
        optimizer = GradientDescent(cost)

        # Train on data
        optimizer.run(net, self.X, self.T)

        # Prediction on new data
        pred = net.predict(self.X)
        pred = np.argmax(pred, 1)

        metrics.roc_auc_score(pred, self.t)