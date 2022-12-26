import sys
import numpy as np
from layers.dense import Dense
from layers.flatten import Flatten
from losses import CategoricalCrossentropy

class NeuralNetwork:

    def __init__(self, layers=None):
        self.layers = layers

    def fit(self, x_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.0):
        """Trains the model on the training data.
        Args:
            x_train - the training data as a Numpy array
            y_train - the training labels as a Numpy array
            epochs - the number of iterations to train the model
            batch_size - the number of samples per batch
            validation_split - fraction of the data to use as validation data
        """


        # populate the weights using the shape of the data, one weight for each column
        if self.layers is not None and len(self.layers) > 1:
            for layer in self.layers:
                if isinstance(layer, Dense):
                    layer.init_weights(x_train)

        # get the length (number of rows) of the data
        l = len(x_train)

        # make sure the the data and the labels are the same length
        if l != len(y_train):
            raise ValueError("x_train and y_train do not have the same length")

        # make sure the classes array is in the right form
        if len(y_train.shape) == 1:
            # convert the classes array to one-hot encoded matrix
            y_train = self.__to_categorical(y_train)

        # initialize the lists to store costs and accuracies
        losses, accuracies = [], []
        val_losses, val_accuracies = [], []

        # start training the model using stochastic gradient descent
        for e in range(1, epochs + 1):

            # get the validation data by splitting the training data
            if validation_split > 0.0:
                # this method shuffles the datasets by default
                x_train, x_test, y_train, y_test = \
                        self.__data_split(x_train, y_train, test_size = validation_split)
            else:
                # otherwise, just shuffle the datasets
                x_train, y_train = self.__shuffle(x_train, y_train)

            # create the batches
            x_train_batch = [x_train[i : i + batch_size] for i in range(0, l, batch_size)]
            y_train_batch = [y_train[i : i + batch_size] for i in range(0, l, batch_size)]

            # progress status
            print(f"Epoch {e}/{epochs}")

            for b in range(len(x_train_batch)):
                # stochastic gradient descent magic
                self.__batch_update_network(x_train_batch[b], y_train_batch[b], l)

                # compute accuracy and loss on the training data
                accuracy = self.__accuracy(x_train, y_train)
                loss = self.__cost(x_train, y_train)
                accuracies.append(accuracy)
                losses.append(loss)

                if validation_split > 0.0:
                    # compute accuracy and loss on the validation data
                    val_accuracy = self.__accuracy(x_test, y_test)
                    val_loss = self.__cost(x_test, y_test)
                    # print training progress
                    self.__progressbar(b, len(x_train_batch) - 1,
                                       loss, accuracy, val_loss, val_accuracy)
                    val_accuracies.append(val_accuracy)
                    val_losses.append(val_loss)
                else:
                    # print progress without the validation accuracy and loss
                    self.__progressbar(b, len(x_train_batch) - 1, loss, accuracy)

            # print an empty line for the progress bar
            print()


    def predict(self, inputs):
        """Passes the activated outputs computed from the inputs from one layer to the
        next for all layers. The outputs of the last layer are the outputs of the network.
        Args:
            inputs - new data for the network to make predictions
        """

        # flatten the inputs into a 1-dimensional array
        outputs = Flatten.call(inputs)

        # iterate through all the layers
        for layer in self.layers:
            # only through the dense layers in the network
            if isinstance(layer, Dense):
                # the outputs of a layer becomes the inputs to the next layer
                outputs = layer.forward(outputs)

        # return outputs of the last layer
        return outputs


    def summary(self):
        """Prints a layer by layer summary of the total number of parameters."""

        print("_" * 64)
        print("Layer Type                Output Shape              Param #   ")
        print("=" * 64)

        # keep count of the total number of parameters
        total_param = 0

        # make sure the network has at least one layer
        if self.layers is not None and len(self.layers) > 0:
            # iterate through all the layers
            for layer in self.layers:
                print(f"{type(layer).__name__}".ljust(26, " "), end = "")
                print(f"(None, {layer.nodes})".ljust(26, " "), end = "")
                if isinstance(layer, Flatten):
                    # flatten layers have no parameters
                    param = 0
                else:
                    # compute the parameters of the dense layers
                    param = np.prod(layer.weights.shape) + layer.nodes
                    total_param += param
                print(f"{param}", end = "\n\n")

        print("=" * 64)
        print(f"Total params: {total_param}")
        print("_" * 64)


    ########################################
    #           PRIVATE HELPERS            #
    ########################################

    def __batch_update_network(self, x_train, y_train, length):
        pass
        # biases_gradient, weights_gradient = self.__backpropagation(x_train, y_train)


    def __backpropagation(self, x, y):
        """Update the weights and biases of the network through backpropagation."""
        pass
        # for layer in self.layers:
            # if isinstance(layer, Dense):
                # biases_gradient = layer.backward(gradient, learning_rate = 0.1)
                # weights_gradient =


    def __shuffle(self, x, y):
        """Shuffles the elements of two arrays of the same length using the same
        random indices.
        @param:
            x - array or matrix to shuffle
            y - array or matrix to shuffle
        @return
            shuffled arrays
        """
        if len(x) != len(y):
            raise ValueError("shuffle: length mismatch")

        # create a range of indices the same size as the datasets
        indices = np.arange(len(x))
        # shuffle the indices
        np.random.shuffle(indices)
        # shuffle the datasets using the shuffled indices
        return x[indices], y[indices]


    def __data_split(self, x, y, test_size = 0.1, shuffle = True):
        """A simple implementation of sklearn.model_selection.train_test_split.
        Split arrays and matrices into random train and test subsets.
        @params:
            x - data array
            y - a corresponding array with the classes/labels
            test_size - the proportion of the dataset to include in the test split
            shuffle - whether or not to shuffle the data before splitting
        @return:
            x_train - training data of proportion (1 - test_size)
            x_test - testing data of proportion (test_size)
            y_train - the corresponding labels as a proportion of (1 - test_size)
            y_test - the corresponding labels as a proportion of (test_size)
        """
        if shuffle:
            # shuffle the datasets using the shuffled indices
            x, y = self.__shuffle(x, y)

        # get the test_size percentage of length as a number
        train_size = int((1 - test_size) * len(x))
        split = np.random.choice(np.arange(len(x)), train_size)

        # split the datasets
        x_train, x_test = x[split], x[~split]
        y_train, y_test = y[split], y[~split]
        return x_train, x_test, y_train, y_test


    def __to_categorical(self, y, num_classes = None, dtype = "float32"):
        """Converts a single class array to a binary class matrix.
        This is also known as one-hot encode and used for categorical crossentropy.
        https://github.com/keras-team/keras/blob/v2.11.0/keras/utils/np_utils.py#L23-L76
        """

        # convert the elements to intengers
        y = np.array(y, dtype = "int")

        # get the shape
        shape = y.shape

        # fix the shape for matrices
        if shape and shape[-1] == 1 and len(shape) > 1:
            shape = tuple(shape[-1])

        # flatten the array
        y.ravel()

        # get the number classes for the binary class matrix
        if not num_classes:
            num_classes = np.max(y) + 1

        # return the binary class matrix
        return np.eye(num_classes, dtype = dtype)[y]


    def __progressbar(self, i, total, loss, accuracy, val_loss = None, val_accuracy = None):
        """Progress bar to display the stats as the model as it trains.
        Args:
            i - current iteration number
            total - total number of iterations
        """
        size = 30 # size of progress bar
        x = int((size * i) / total) # i as a percentage of size
        s = f"loss: {loss:.3f} - accuracy: {accuracy:.3f}"
        if val_loss is not None and val_accuracy is not None:
            s = f"{s} - val_loss: {val_loss:.3f} - val_accuracy: {val_accuracy:.3f}"
        print(f"{i}/{total} [{'='*x}{'.'*(size-x)}] - {s}",
              end = "\r", file = sys.stdout, flush = True)

    def __accuracy(self, x_test, y_test):
        """Compute the accuracy score of the model."""

        # get the predictions
        y_pred = self.predict(x_test)

        # y_pred = self.__to_categorical([np.argmax(p) for p in y_pred], num_classes = y_test[0].size)

        # convert the predictions to binary class matrix
        one_hot = []
        for pred in y_pred:
            for y in pred:
                encoded = self.__to_categorical(np.argmax(y), num_classes = y_test[0].size)
                one_hot.append(encoded)

        one_hot = np.array(one_hot)

        # sum up all the correct predictions
        correct_pred = 0
        for p, t in zip(y_pred, y_test):
            correct_pred += int(np.array_equal(p, t))

        # divide all the correct predictions by the total
        return correct_pred / len(y_pred)

    def __cost(self, x_test, y_true, loss = CategoricalCrossentropy):
        """Compute the loss score of the model."""

        # get the predictions
        y_pred = self.predict(x_test)

        # convert the predictions to binary class matrix
        y_pred = self.__to_categorical([np.argmax(p) for p in y_pred],
                                       num_classes = y_true[0].size)

        # return the loss score
        return loss.compute(y_pred, y_true) / len(x_test)


    def __weights_and_biases(self):
        """Retrieves the weights and biases from all the layers."""
        weights = np.array([])
        biases = np.array([])
        for layer in layers:
            if isinstance(layer, Dense):
                self.weights = np.append(weights, layer.weights)
                self.biases = np.append(biases, layer.biases)
        return weights, biases
