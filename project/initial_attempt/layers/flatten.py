import numpy as np
from layers.layer import Layer

class Flatten(Layer):

    def __init__(self, input_shape=None):
        """Args:
            input_shape - a tuple with the shape of the data
        Note:
            The output size of the flatten layer is only named 'nodes' for consistency with
            the other layers.
        """
        # compute the output size of the flatten layer
        self.nodes = np.prod(input_shape)

    def call(inputs):
        """Converts an n-dimension array into a 1-dimensional numpy array.
        For example, if the inputs shape is (10, 28, 28) then it returns
        an a numpy array of shape (10, 784), where 10 represents the indices.
        Args:
            inputs - a Numpy array to flatten
        """
        # create list to store flattened inpus
        flattened_inputs = []

        for i in range(len(inputs)):
            # flatten the inputs matrix into a 1D array in row-major order
            flattened_inputs.append(inputs[i].flatten(order = "C"))

        # return the flattened_inputs as a numpy array
        return np.array(flattened_inputs)
