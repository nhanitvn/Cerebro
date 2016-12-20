import numpy as np

class Initializer(object):
    pass


class Uniform(Initializer):
    @staticmethod
    def init(shape):
        return np.random.randn(*shape)

class GlorotUniform(Initializer):
    def init(self, weights):
        pass
