from layers.layer import Layer

class Activation(Layer):

    def __init__(self, activation, activation_prime):
        self.__activation = activation
        self.__activation_prime = activation_prime

    def forward(self, inputs):
        self.__inputs = inputs
        return self.__activation(inputs)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.__activation_prime(self.__inputs))
