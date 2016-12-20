import numpy as np


class Layer(object):
    def __init__(self, size):
        self._size = size

    @property
    def size(self):
        return self._size

    def initialize(self, net):
        self._net = net

    def forward(self, input_val, epsilon=0):
        pass

    def backward(self, gradient):
        pass

    def update_weights(self, learning_rate):
        pass


class FeedForward(Layer):
    def __init__(self, size, activation, initializer):
        super(FeedForward, self).__init__(size)
        self._activation = activation
        self._input_size = 0
        self._input_val = None
        self._initializer = initializer

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

    def initialize(self, net):
        super(FeedForward, self).initialize(net)
        self._weights = self._initializer.init((net.last_layer.size, self._size))
        self._bias = self._initializer.init((1, self._size))

    def forward(self, input_val, epsilon=0):
        self._input_val = input_val
        return self._activation.forward(input_val.dot(self._weights + epsilon) + (self._bias + epsilon))

    def backward(self, gradient):
        output_gradient = self._activation.backward(gradient)
        self._gradient = self._input_val.T.dot(output_gradient)
        return output_gradient.dot(self._weights.T)

    def update_weights(self, learning_rate):
        # Update this layer's weights
        self._weights -= learning_rate * self._gradient
        self._bias -= learning_rate * np.ones(self._bias.shape)


class Input(Layer):
    def __init__(self, size):
        super(Input, self).__init__(size)

    def forward(self, input_val):
        self._output = input_val
        return self._output

    def backward(self, gradient):
        self._gradient = gradient
        return self._gradient
