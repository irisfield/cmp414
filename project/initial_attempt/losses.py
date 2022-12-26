import numpy as np

class Loss:

    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        """Compute the mean loss.
        Args:
            y_pred - the predicted labels are the output from the model
            y_true - the true labels are the intended target values
        """
        sample_losses = self.__call(y_pred, y_true)
        batch_loss = np.mean(sample_losses)
        return batch_loss

    def __call(self, y_pred, y_true):
        pass

class CategoricalCrossentropy:
    """Computes the crossentropy loss between the labels and predictions.
    This class handles both integer and one-hot enconded labels."""

    def compute(y_pred, y_true):
        """Compute the Categorical Crossentropy loss.
        This class expects classes as a binary class matrix / one-hot enconded.
        Args:
            y_pred - the predicted labels (the outputs of the last layer)
            y_true - the true labels from data
        """

        if len(y_true.shape) > 2:
            raise Exception(f"unsupported shape: {len(y_true.shape)}")


        """Clip the predicted labels to avoid infinity and division by zero
        errors when computing the negative natural log and the mean of the losses.
        The clipping threshold should be a number very close to zero but not zero."""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # get the the confidences of the predictions
        # multiply each row element-wise and then summing up each row
        confidences = np.sum(y_pred * y_true[:10], axis = 1)

        # losses
        sample_losses = -np.log(confidences)

        # return the categorical crossentropy loss
        return np.mean(sample_losses)


    def error(z, y_pred, y_true):
        """Compute the error from the output layer."""
        return y_pred - y_true
