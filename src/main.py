import numpy as np
import data_loader
import backprop


def initialize_weights():
    pass


if __name__ == '__main__':
    # Load training data
    input_x_features = data_loader.load_training_data_from_binary()
    input_labels = data_loader.load_training_labels_from_binary()

    # Load testing data
    testing_features = data_loader.load_testing_data_from_binary()
    testing_labels = data_loader.load_testing_labels_from_binary()

    # Train the neural net
    backprop.do_train(input_x_features, data_loader.transform_output_label_into_vector(input_labels))

    # Test the neural net


class ParameterFactory:
    _num_hidden_units = None
    _mini_batch_size = None
    _learning_rate = None
    _num_output_units = 10

    _training_unit_weights = None
    _hidden_unit_weights = None

    def __init__(self, num_hidden_units=15, mini_batch_size=100, learning_rate=0.1):
        self._learning_rate = learning_rate
        self._mini_batch_size = mini_batch_size
        self._num_hidden_units = num_hidden_units

    def num_hidden_unit(self):
        return self._num_hidden_units

    def mini_batch_size(self):
        return self._mini_batch_size

    def learning_rate(self):
        return self._learning_rate

    def num_output_unit(self):
        return self._num_output_units

    def initialize_weights(self, num_input_units=51, num_output_units=10):
        if self._training_unit_weights is None:
            training_unit_weights = np.full((num_input_units, self._num_hidden_units), 0.5)
        else:
            training_unit_weights = self._training_unit_weights

        if self._hidden_unit_weights is None:
            hidden_unit_weights = np.full((self._num_hidden_units, num_output_units), 0.5)
        else:
            hidden_unit_weights = self._hidden_unit_weights

        return training_unit_weights, hidden_unit_weights
