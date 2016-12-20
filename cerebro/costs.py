import numpy as np


class Cost(object):
    pass


class CrossEntropy(Cost):
    def compute_cost(self, output, target):
        return - np.multiply(target, np.log(output)).sum()

    def compute_gradient(self, output, target):
        return - target / output


class SquaredLoss(Cost):
    def comput_cost(self, output, target):
        return 0.5 * (target - output) ** 2

    def compute_gradient(self, output, target):
        return target - output