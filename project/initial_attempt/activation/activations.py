import numpy as np
from layers.layer import Layer
from activation.activation import Activation

class ReLU(Activation):
    """Rectified Linear Unit activation function for hidden layers."""

    def __init__(self):

        # the relu function
        self.__relu = lambda x: max(0, x)

        super().__init__(self.__relu, self.__relu)

class Sigmoid(Activation):

    def __init__(self):

        # the sigmoid function
        self.__sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))

        # the derivative of the sigmoid function
        self.__sigmoid_prime = lambda z: self.__sigmoid(z) * (1 - self.__sigmoid(z))

        super().__init__(self.__sigmoid, self.__sigmoid_prime)

class Softmax(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """Softmax activation function.
        Args:
            inputs - the inputs to convert into probability distributions
        Note:
            axis = 1 : row-wise operation
            keepdims = True : retain the original dimension after operation
        """

        # protects against overflow errors when exponentiating without affecting results
        overflow_protected = inputs - np.max(inputs, axis = 1, keepdims = True)

        # exponentiate the inputs using Euler's number
        exponentiated = np.exp(overflow_protected)

        # normalize them by taking the inputs and dividing them by the sum of their total
        self.__outputs = exponentiated / np.sum(exponentiated, axis = 1, keepdims = True)

        return self.__outputs

    def backward(self, output_gradient):
        n = np.size(self.__outputs)
        return np.dot((np.identity(n) - self.__outputs.T) * self.__outputs, output_gradient)
