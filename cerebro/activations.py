import numpy as np
from cerebro.layers import Layer


class Activation(Layer):
    pass


class ReLU(Activation):
    def __init__(self):
        self.output = None

    def forward(self, input_val):
        flatten_input_val = np.reshape(input_val, input_val.size)
        max_val = np.maximum(0, flatten_input_val)
        self._output = np.reshape(max_val, input_val.shape)
        return self._output

    def backward(self, gradient):
        self._gradient = np.sign(self._output) * gradient
        return self._gradient


class Softmax(Activation):
    def __init__(self):
        self._output = None

    def forward(self, input_val):

        exp = np.exp(input_val)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        self._output = exp / sum_exp
        return self._output

    def backward(self, gradient):
        # http://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
        tmp = gradient * self._output
        s_tmp = np.sum(tmp, axis=tmp.ndim - 1, keepdims=True)
        self._gradient = tmp - self._output * s_tmp
        return self._gradient
