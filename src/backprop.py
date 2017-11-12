import numpy as np
from main import ParameterFactory
import scipy.special as special



def feed_forward(training_example,
                 input_layer_weights,
                 hidden_layer_weights,
                 parameter_factory=None):
    """
    Feed forward part of the
    :rtype: object
    :param training_example: 51 x 1 ndarray
    :param input_layer_weights: num_hidden_units x 51 ndarray
    :param hidden_layer_weights:  num_output_units(10) x num_hidden_units ndarray
    """
    num_input_units, num_hidden_units = input_layer_weights.shape
    assert training_example.shape[0] == num_input_units

    # calculate values at hidden layer
    hidden_layer_values = special.expit(np.dot(training_example, input_layer_weights))

    assert parameter_factory.num_hidden_unit() == hidden_layer_values.shape[0]
    print('Hidden layer shape: {}'.format(hidden_layer_values.shape))

    # calculate values at output layer
    output_layer_values = special.expit(np.dot(hidden_layer_values, hidden_layer_weights))
    assert 10 == output_layer_values.shape[0]
    print('Output layer shape: {}'.format(output_layer_values.shape))

    return hidden_layer_values, output_layer_values


def output_unit_error(output_unit_value, actual_value):
    return output_unit_value * (1.0 - output_unit_value) * (actual_value - output_unit_value)


def back_propagate_errors(training_example_features,
                          training_example_label,
                          training_layer_weights,
                          hidden_layer_weights,
                          hidden_layer_values,
                          output_layer_values,
                          parameter_factory=None):
    # Calculate error for each output unit
    output_error_func = np.vectorize(output_unit_error)
    output_unit_error_terms = output_error_func(output_layer_values, training_example_label)

    # Calculate error for each hidden unit

    # Update weights


def do_train(training_data_features, training_data_labels):
    """

    :param training_data_features: 60000 x 50 matrix
    :param training_data_labels:  60000 x 1 matrix
    """
    assert training_data_features.shape[0] == training_data_labels.shape[0]

    parameter_factory = ParameterFactory()
    training_unit_weights, hidden_unit_weights = parameter_factory.initialize_weights()

    # Add bias units to the training features. This makes it 51 units in the input layer with the
    # first input unit being of fixed value 1.0
    training_data_features_with_bias = np.insert(training_data_features, 0, np.full((60000,), 1.0), axis=1)


    for idx, training_data in enumerate(training_data_features_with_bias):
        hidden_layer_values, output_layer_values = feed_forward(training_data,
                                                                training_unit_weights,
                                                                hidden_unit_weights,
                                                                parameter_factory)
        back_propagate_errors(training_data,
                              training_data_labels[idx],
                              training_unit_weights,
                              hidden_unit_weights,
                              hidden_layer_values,
                              output_layer_values,
                              parameter_factory)
