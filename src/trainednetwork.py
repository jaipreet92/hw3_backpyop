import numpy as np
import scipy.special as special


class TrainedNetwork:
    _input_layer_weights = None
    _hidden_layer_weights = None

    def __init__(self, input_layer_weights, hidden_layer_weights):
        self._input_layer_weights = input_layer_weights
        self._hidden_layer_weights = hidden_layer_weights

    def _least_square_error(self, testing_label, output_units):
        assert testing_label.shape[0] == output_units.shape[0]
        return np.exp2(output_units - testing_label)

    def get_weights(self):
        return self._input_layer_weights, self._hidden_layer_weights

    def test_predictions(self, testing_features, testing_labels):
        # Defensive checks
        if self._input_layer_weights is None or self._hidden_layer_weights is None:
            raise ValueError('Weights not initialized')
        assert testing_labels.shape[1] == 10
        num_examples, num_features = testing_features.shape
        assert num_features == self._input_layer_weights.shape[0]

        for i in range(num_examples):
            hidden_unit_outputs = special.expit(np.dot(testing_features[i], self._input_layer_weights))
            output_units = special.expit(np.dot(hidden_unit_outputs, self._hidden_layer_weights))
            self._least_square_error(testing_labels[i], output_units)

            predicted_value = np.argmax(output_units)
            actual_value = np.argmax(testing_labels[i])
            print('Output for sample number {} Predicted Value: {} , Actual Value: {}'.format(i, predicted_value,
                                                                                              actual_value))
            # get prediction from output units
            # get Mean Square error
        return 0.0

    def test_prediction(self, testing_example, output):
        pass
