from cerebro.layers import Layer


class Cerebro(Layer):
    def __init__(self):
        self._layers = []

    @property
    def layers(self):
        return self._layers

    @property
    def last_layer(self):
        return self._layers[-1]

    def forward(self, X):
        output = X
        for layer in self._layers:
            output = layer.forward(output)
        return output

    def backward(self, gradient):
        backward_gradient = gradient
        for layer in reversed(self._layers):
            backward_gradient = layer.backward(backward_gradient)

    def update_weights(self, learning_rate):
        for layer in self._layers:
            layer.update_weights(learning_rate)
            

    def add(self, layer):
        # TODO: Check for Layer type
        layer.initialize(self)
        self._layers.append(layer)

    def predict(self, X):
        return self.forward(X)
