
class Optimizer(object):
    def __init__(self):
        pass

    def run(self, net, data):
        pass


class GradientDescent(Optimizer):
    def __init__(self, cost, num_epochs=10, learning_rate=0.001):
        self._cost = cost
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate

    def run(self, net, X, y):
        for i in range(self._num_epochs):
            # Do forward
            output = net.forward(X)

            # Compute and output cost
            print('Loss at epoch {}: {}'.format(i+1, self._cost.compute_cost(output, y)))

            # Compute cost's gradient
            gradient = self._cost.compute_gradient(output, y)

            # Do back-propagation
            net.backward(gradient)

            # Update weights
            net.update_weights(self._learning_rate)

