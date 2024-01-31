import numpy as np

class Neuron():
    def __init__(self, dendrites=1, activation='sigmoid'):
        self.dendrites = dendrites
        self.weights = np.ones(dendrites + 1)

    def set_size(self, size):
        self.weights = np.ones(size + 1)
        self.dendrites = size

    def activate(self, inputs):
        if len(inputs) != self.dendrites:
            raise ValueError()
        result_activate = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 / (1 + np.exp(-result_activate))  # Sigmoid activation function


class Input():
    def __init__(self, input_size=1):
        self.input_size = input_size
        self.output_size = input_size

    def activate(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError()
        return inputs


class Dense():
    def __init__(self, neurons=1, input_size=1, activation='sigmoid'):
        self.neurons = neurons
        self.input_size = input_size
        self.output_size = neurons
        self.activation = activation
        self.layer = [Neuron(dendrites=input_size) for _ in range(neurons)]

    def set_size(self, input_size):
        self.input_size = input_size
        self.layer = [Neuron(dendrites=input_size) for _ in range(self.neurons)]

    def activate(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError()
        outputs = [neuron.activate(inputs) for neuron in self.layer]
        return outputs


class Sequential():

    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        if self.layers:
            layer.set_size(self.layers[-1].output_size)
        self.layers.append(layer)

    def predict(self, inputs):
        signal = inputs
        for layer in self.layers:
            signal = layer.activate(signal)
        return signal

# Example usage:
model = Sequential(
    Input(input_size=3),
    Dense(neurons=3),
    Dense(neurons=2)
)

output = model.predict([1, 1, 1])
print(output)
