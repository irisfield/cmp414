import sys
import numpy as np

class NeuralNetwork:

    def __init__(self, input_shape = None):
        # the weights are based on the shape of the data
        self.weights = None
        # pick a random bias from the standard distribution
        self.bias = np.random.randn()

    def _init_weights(self, x_train):
        """Initialize the weights based on the shape of the training data."""

        # the training data is expected as 2D array for setting the weights
        if len(x_train.shape) == 1:
            # if it is 1D, make it a 2D array
            x_train = x_train.reshape(len(x_train), 1)

        # n-dimensional arrays are not supported
        if len(x_train.shape) > 2:
            # the training is expceted to be a 1D or 2D array
            raise ValueError("this neural network only supports data as 1D or 2D arrays")

        # one weight for each column in the array
        num_weights = x_train.shape[-1]
        # initialize the weights using He Normal weight initialization
        self.weights = np.random.randn(num_weights) * np.sqrt(2 / num_weights)

    def _sigmoid(self, x):
        """The sigmoid function converts input into a probability distribution.
        The sigmoid function always returns a value between 0 and 1.
        @param
            x - input array
        @return
            x as a probability distribution
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        """The derivative of the sigmoid function."""
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _compute_gradients(self, x_sample, y_sample):
        """Apply stochastic gradient descent using backpropagation.
        This function computes the partial derivatives and applies the chain rule.
        @return
            the derivatives for the weights and bias"""

        # get the prediction for the sample
        output = np.dot(x_sample, self.weights) + self.bias
        prediction = self._sigmoid(output)

        # compute the derivative of the error with respect to the prediction
        derror_dprediction = 2 * (prediction - y_sample)
        # compute the derivative of the prediction with respect to the output
        dprediction_doutput = self._sigmoid_prime(output)

        # compute the derivative of the output with respect to the bias
        doutput_dbias = 1
        # compute the derivative of the output with respect to the weights
        doutput_dweights = (0 * self.weights) + (1 * x_sample)

        # compute the derivative of the error with respect to bias
        derror_dbias = derror_dprediction * dprediction_doutput * doutput_dbias
        # compute the derivative of the error with respect to the weights
        derror_dweights = derror_dprediction * dprediction_doutput * doutput_dweights
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights, learning_rate):
        """Subtracts the gradients times the learning rate from the weights and bias."""
        self.bias -= (derror_dbias * learning_rate)
        self.weights -= (derror_dweights * learning_rate)

    def _progressbar(self, i, total, mse):
        """Progress bar to display the stats as the model as it trains.
        Args:
            i - current iteration number
            total - total number of iterations
        """
        size = 40 # size of progress bar
        x = int((size * i) / total) # i as a percentage of size
        print(f"{i:0{len(str(total))}}/{total} [{'='*x}{'-'*(size-x)}] - mse: {mse:.2f}",
              end = "\r", file = sys.stdout, flush = True)

    def _mse(self, x_test, y_test, epoch, epochs):
        cumulative_mse = 0
        # compute the mean mean squared error for each sample
        for i in range(len(x_test)):
            # get the sample data points
            x, y = x_test[i], y_test[i]
            # compute the mean squared error
            prediction = self.predict(x)
            cumulative_mse += np.square(prediction - y)

        # display the training progress
        self._progressbar(epoch, epochs, cumulative_mse)

    def predict(self, inputs):
        """Makes prediction from the inputs.
        @params
            inputs - array of data points
        @return
            prediction
        """
        # compute output from the inputs
        output = np.dot(inputs, self.weights) + self.bias
        # convert the output to a probability distribution
        prediction = self._sigmoid(output)

        if len(prediction.shape) == 1:
            # make the predictions into a 2D array to display them vertically
            prediction = prediction.reshape([len(prediction), 1])

        return prediction

    def fit(self, x_train, y_train, epochs = 10, learning_rate = 0.01):
        """Trains the model on the training data.
        @params:
            x_train - array of data points to use for training
            y_train - the labels corresponding to the data points
            epochs - the number of iterations to train
        """

        # ensure the data are numpy arrays
        x_train = np.array(x_train, dtype = "float32")
        y_train = np.array(y_train, dtype = "float32")

        # initialize the weights to random values from the standard distribution
        self._init_weights(x_train)

        # start training the model using stochastic gradient descent
        for epoch in range(epochs):

            # select a random index
            random_index = np.random.randint(len(x_train))
            # pick a random sample from the training data
            x_sample = x_train[random_index]
            # pick the corresponding class/label for the sample
            y_sample = y_train[random_index]
            # compute the partial derivatives
            derror_dbias, derror_dweights = self._compute_gradients(x_sample, y_sample)
            # update the weights and biases using the gradients
            self._update_parameters(derror_dbias, derror_dweights, learning_rate)

            # compute the mean squared error and show the training progress
            if epoch % 100 == 0:
                self._mse(x_train, y_train, epoch, epochs)
                print()
            if epoch == epochs - 1:
                self._mse(x_train, y_train, epoch + 1, epochs)


    ######################################################
    # OTHER HELPER METHODS (UNIMPLEMENTED FUNCTIONALITY) #
    ######################################################

    def _binary_cross_entropy(self, y_true, y_pred):
        """Computes the loss."""
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def _binary_cross_entropy_prime(self, y_true, y_pred):
        """The derivating of binary cross entropy."""
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    def _cost(self, x_test, y_test, reg = 0.1):
        """Compute the loss/cost of
        @params:
            x_test - a data point
            y_test - the corresponding class/label for x_test
            reg - regularization parameter
        @return
            loss score
        """
        y_pred = self.predict(x_test)
        loss = self._binary_cross_entropy_prime(y_test, y_pred)
        loss += 0.5 * (reg / len(x_test)) * \
                sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return loss

    def _accuracy(self, x_test, y_test):
        """Compute the accuracy score of the model.
        Accuracy is only supported for binary classes or one-hot encoded."""

        # make sure that the labels are binary classes
        if not np.all([n == 0 or n == 1 for n in np.unique(y_test)]):
            raise ValueError("only binary or one-hot encoded classes are supported")

        # ensure the binary classes are in a numpy array
        y_test = np.array(y_test, dtype = "int64")
        # get the predictions
        y_pred = self.predict(x_test)
        # convert the predictions into binary classes
        y_pred = (y_pred > 0.5) * 1
        # count the number of correct predictions
        correct_pred = np.count_nonzero(y_pred == y_test)
        # divide all the correct predictions by the total
        return correct_pred / len(y_pred)

    def _shuffle(self, x, y):
        """Shuffles the elements of two arrays of the same length using the same
        random indices.
        @param:
            x - array or matrix to shuffle
            y - array or matrix to shuffle
        @return
            shuffled arrays
        """
        if len(x) != len(y):
            raise ValueError("shuffle: array lengths do not match")

        # create a range of indices the same size as the datasets
        indices = np.arange(len(x))
        # shuffle the indices
        np.random.shuffle(indices)
        # shuffle the datasets using the shuffled indices
        return x[indices], y[indices]


    def _data_split(self, x, y, test_size = 0.1, shuffle = True):
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
