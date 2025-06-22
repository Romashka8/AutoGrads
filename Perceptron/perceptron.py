import numpy as np

import interface

# ----------------------------------------------------------------------------------------------------------------------------

activation_func = {
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'relu': lambda x: np.max(x, 0),
    'tanh': lambda x: np.tanh(x),
    'linear': lambda x: x, # оставим для линейного слоя, дабы не нарушать структуры
}

activation_der = {
    'sigmoid': lambda x: x * (1 - x), # под x понимаем значение сигмоиды при прямом проходе
    'relu': lambda x: np.where(x > 0, 1, 0),
    'tanh': lambda x: 1 - np.tanh(x) ** 2,
    'linear': lambda x: np.ones_like(x), # оставим для линейного слоя, дабы не нарушать структуры
}

# ----------------------------------------------------------------------------------------------------------------------------

class Neuron(interface.Neuron):

    def __init__(self, input_size, func='sigmoid'):
        
        self.weights = np.random.rand(input_size + 1) * 0.1
        self.func = activation_func[func]
        self.der = activation_der[func]
        self.output = 0
        
    def activate(self, x):
        
        z = np.dot(x, self.weights[:-1]) + self.weights[-1]
        self.output = self.func(z)
        
        return self.output

# ----------------------------------------------------------------------------------------------------------------------------

class Layer(interface.Layer):

    def __init__(self, num_neurons, input_size, func='sigmoid'):

        self.neurons = [Neuron(input_size, func) for _ in range(num_neurons)]
        self.output = 0
        self.error = 0
        self.delta = 0

    def forward(self, x):
        
        self.output = np.array([neuron.activate(x) for neuron in self.neurons])
        return self.output

# ----------------------------------------------------------------------------------------------------------------------------

class Network(interface.Network):

    def __init__(self, layers):

        self.layers = layers

    def forward(self, x):

        for layer in self.layers: x = layer.forward(x)
        
        return x

    def backward(self, x, y, learning_rate):

        # обновим данные выходного слоя
        output = self.forward(x)
        error = y - output
        
        self.layers[-1].error = error
        self.layers[-1].delta = error * self.layers[-1].neurons[0].der(output)

        # распространим ошибку по всей сети
        for i in reversed(range(len(self.layers) - 1)):

            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            layer.error = np.dot(next_layer.delta, np.array([neuron.weights[:-1] for neuron in next_layer.neurons]))
            layer.delta = layer.error * np.array([neuron.der(neuron.output) for neuron in layer.neurons])

        # обновление весов
        for i in range(len(self.layers)):

            layer = self.layers[i]
            inputs = x if i == 0 else self.layers[i - 1].output

            for j, neuron in enumerate(layer.neurons):

                for k in range(len(neuron.weights) - 1):

                    neuron.weights[k] += learning_rate * layer.delta[j] * inputs[k]

                neuron.weights[-1] += learning_rate * layer.delta[j]

    def train(self, x, y, learning_rate, epochs):

        for epoch in range(epochs):

            for xi, yi in zip(x, y):

                self.backward(xi, yi, learning_rate)

# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	x = np.array([
	    [0.12, 0.34],
	    [0.25, 0.41],
	    [0.31, 0.29],
	    [0.47, 0.5],
	    [0.53, 0.38],
	    [0.6, 0.45],
	    [0.72, 0.21],
	    [0.84, 0.36],
	    [0.91, 0.52],
	    [0.1, 0.4]
	])

	# Целевая переменная y
	y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

	input_size = 2
	hidden_size_1 = 5
	hidden_size_2 = 4
	output_size = 1

	layer_1 = Layer(hidden_size_1, input_size, 'relu')
	layer_2 = Layer(hidden_size_2, hidden_size_1, 'relu')
	layer_3 = Layer(output_size, hidden_size_2, 'sigmoid')

	network = Network([layer_1, layer_2, layer_3])

	learning_rate = 0.1
	epochs = 10000

	network.train(x, y, learning_rate, epochs)

	for xi in x:
	    output = network.forward(xi)
	    print('Input: ', xi, 'Output: ', int(output > 0.5))
