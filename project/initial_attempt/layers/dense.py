"""Saving a model means saving the weights and biases.
Loading a model means getting the weights and biases from the saved model.
When making a new neural network, it is neccessary to initialize the weights and biases.

The biases are typically just initialized to just zero, but there are some cases where
that may not be ideal. For example, if the inputs times the weights is not big enough
to produce an output, using a bias of zero means that the neuron is just going to output
zero. Then that zero becomes the input to the next neuron, and following that pattern,
the zeros propagate throughout the whole network. This is an example of a "dead" network,
where it only outputs zeros. In this case, initializing the biases to non-zero will fix it.
"""

import numpy as np
from layers.layer import Layer
from losses import CategoricalCrossentropy

class Dense(Layer):
    def __init__(self, nodes, activation=None, loss = CategoricalCrossentropy):
        """Args:
            input_shape - a tuple with the shape of the data
            nodes - the number of nodes (and biases) to use
        """

        self.nodes = nodes
        self.weights = None

        # pass in a tuple of the shape of the biases: 1 by the number of neurons in the layer
        self.biases = np.random.randn(nodes, 1)

        # variables to store the last outputs of the layer
        self.__outputs = None

        # activation function to use
        self.__activation = activation

        # cost function to compute the loss
        self.__loss = loss

    def init_weights(self, x_train):
        """Initialize the weights based on the shape of the training data."""

        # the training data is expected as 3D array for setting the weights
        if len(x_train.shape) == 1:
            # if it is 1D, make it a 3D array
            x_train = x_train.reshape(len(x_train), 1, 1)

        if len(x_train.shape) == 2:
            # if it is 2D, make it a 3D array
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        # n-dimensional arrays are not supported
        if len(x_train.shape) > 3:
            # the training is expceted to be a 2D or 3D array
            raise ValueError("this neural network only supports data as 2D or 3D arrays")

        # one weight for each column in the array
        num_weights = np.prod(x_train.shape[-2:])
        # initialize the weights using He Normal weight initialization
        self.weights = np.random.randn(num_weights) * np.sqrt(2 / num_weights)

    def forward(self, inputs):
        """Computes the outputs from the inputs, applies the activation function,
        and returns the activated outputs so it can passed to the next layer."""

        # compute the outputs matrix for the first sample
        # self.__outputs = np.dot(self.weights, inputs[0]) + self.biases

        # list to store all the dot products
        # for i in range(1, len(inputs)):
            # compute the outputs matrix for each sample
            # sample_outputs = np.dot(self.weights, inputs[i]) + self.biases

            # add the outputs matrix of all the samples
            # self.__outputs = np.add(self.__outputs, sample_outputs)

        # take the average of the outputs of all the samples
        # self.__outputs = self.__outputs / len(inputs)

        # apply the sigmoid function to convert the outputs into probabilities
        # self.__outputs = self.__sigmoid(self.__outputs)

        self.__outputs = []

        for sample in inputs:
            sample_outputs = np.dot(self.weights, sample) + self.biases
            # apply the sigmoid function to convert the outputs into probabilities
            self.__outputs.append(self.__sigmoid(sample_outputs))

        print(sample_outputs.shape)
        # convert list to numpy array
        self.__outputs = np.array(self.__outputs)

        # apply the activation function if any and return results
        if self.__activation == "relu":
            return self.__relu(self.__outputs)
        elif self.__activation == "softmax":
            return self.__softmax(self.__outputs)
        else:
            return self.__outputs

    def backward(self, x_train, y_train):
        """Apply gradient descent using backpropagation."""

        # create the gradients
        biases_gradient = [np.zeros(b.shape) for b in self.biases]
        weights_gradient = [np.zeros(w.shape) for w in self.weights]

        # list of activations
        zs, activations = np.array([]), np.array([])

        # pass forward
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, x_train) + bias
            zs = np.append(zs, z)
            activations = np.append(activations, self.__activation(z))

        # pass backward
        error = self.__loss.error(zs[-1], activations[-1], y_train)
        biases_gradient[-1] = error
        weights_gradient[-1] = np.dot(error, activations[-2].T)

        # update parameters and return input gradient
        weights_gradient = np.dot(output_gradient, self-input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


    ########################################
    #         ACTIVATION FUNCTIONS         #
    ########################################


    def __sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))


    def __sigmoid_prime(self, z):
        """The deriviate of the sigmoid function."""
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))

    def __relu(self, x):
        """Rectified Linear Unit activation function for hidden layers."""
        return np.maximum(0, x)

    def __softmax(self, inputs):
        """Softmax activation function.
        Args:
            inputs - the inputs to convert into probability distributions
        Note:
            axis = 1 : row-wise operation
            keepdims = True : retain the original dimension after operation
        """

        # protect againsts overflow when exponentiating without affecting the results
        overflow_protected = inputs - np.max(inputs, axis = 1, keepdims = True)

        # exponentiate the inputs using Euler's number
        exponentiated = np.exp(overflow_protected)

        # normalize by taking the inputs and dividing them by the sum of their total
        return exponentiated / np.sum(exponentiated, axis = 1, keepdims = True)
