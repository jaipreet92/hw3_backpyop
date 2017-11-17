import numpy as np


class HyperParameters:
    _num_hidden_units = None
    _mini_batch_size = None
    _learning_rate = None
    _num_output_units = 10
    _num_epochs = None
    _momentum = None

    _training_unit_weights = None
    _hidden_unit_weights = None

    def __init__(self, num_hidden_units=100,
                 mini_batch_size=100,
                 learning_rate=0.1,
                 num_epochs=1,
                 num_input_units=51,
                 num_output_units=10,
                 momentum=0.1):
        self._learning_rate = learning_rate
        self._mini_batch_size = mini_batch_size
        self._num_hidden_units = num_hidden_units
        self._num_epochs = num_epochs
        self._num_input_units = num_input_units
        self._num_output_units = num_output_units
        self._momentum = momentum

    def momentum(self):
        return self._momentum

    def num_epochs(self):
        return self._num_epochs

    def num_hidden_unit(self):
        return self._num_hidden_units

    def mini_batch_size(self):
        return self._mini_batch_size

    def learning_rate(self):
        return self._learning_rate

    def num_output_unit(self):
        return self._num_output_units

    def num_input_units(self):
        return self._num_input_units

    def initialize_weights(self):
        if self._training_unit_weights is None:
            training_unit_weights = 1.0 * np.random.random_sample((self._num_input_units, self._num_hidden_units)) - 0.5
        else:
            training_unit_weights = self._training_unit_weights

        if self._hidden_unit_weights is None:
            hidden_unit_weights = 1.0 * np.random.random_sample((self._num_hidden_units, self._num_output_units)) - 0.5
        else:
            hidden_unit_weights = self._hidden_unit_weights

        return training_unit_weights, hidden_unit_weights
