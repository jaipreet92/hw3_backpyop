import numpy as np
from main import ParameterFactory
import scipy.special as special
import time


def get_unit_output(weights, values):
    assert len(weights) == len(values)

    output = 0.0
    for idx, weight in enumerate(weights):
        output += weight * values[idx]
    return output


def feed_forward(training_example,
                 input_layer_weights,
                 hidden_layer_weights,
                 parameter_factory=None):
    """
    Feed forward part of the the BackPropagation algorithmn
    :rtype: object
    :param training_example: 51 x 1 ndarray
    :param input_layer_weights: 51 x num_hidden_units ndarray
    :param hidden_layer_weights: num_hidden_units x 10 ndarray
    """
    num_input_units, num_hidden_units = input_layer_weights.shape
    assert training_example.shape[0] == num_input_units

    # calculate values at hidden layer
    hidden_layer_values = []
    input_layer_weights = input_layer_weights.transpose()  # 15 x 51
    rows, columns = input_layer_weights.shape
    for i in range(rows):
        hidden_unit_value = 0.0
        for j in range(columns):
            hidden_unit_value += training_example[j] * input_layer_weights[i, j]
        hidden_layer_values.append(hidden_unit_value)
    hidden_layer_values = special.expit(hidden_layer_values)
    assert parameter_factory.num_hidden_unit() == hidden_layer_values.shape[0]
    print('Hidden layer shape: {}'.format(hidden_layer_values.shape))

    # calculate values at output layer
    output_layer_values = []
    hidden_layer_weights = hidden_layer_weights.transpose()  # 10 x 15
    rows, columns = hidden_layer_weights.shape
    for i in range(rows):
        output_unit_value = 0.0
        for j in range(columns):
            output_unit_value += hidden_layer_values[j] * hidden_layer_weights[i, j]
        output_layer_values.append(output_unit_value)
    output_layer_values = special.expit(output_layer_values)
    assert 10 == output_layer_values.shape[0]
    print('Output layer shape: {}'.format(output_layer_values.shape))

    return hidden_layer_values, output_layer_values


# def output_unit_error(output_unit_value, actual_value):
#     return output_unit_value * (1.0 - output_unit_value) * (actual_value - output_unit_value)
#
#
# output_error_func = np.vectorize(output_unit_error)
#
#
# def hidden_unit_error(hidden_unit_value, hidden_unit_weights, output_unit_errors):
#     return hidden_unit_value * (1.0 - hidden_unit_value) * np.dot(hidden_unit_weights, output_unit_errors)
#
#
# hidden_unit_error_func = np.vectorize(hidden_unit_error)


def back_propagate_errors(training_example_label,
                          hidden_layer_weights,
                          hidden_unit_values,
                          output_unit_values):
    # Calculate error for each output unit
    # Compare each labels [0 0 0 1 0 0 0 0 0 0] for 3 with the output layer value
    """

    :param training_example_label: 10 x 1 ndarray for actual output layer or label for current training example
    :param hidden_layer_weights: num_hidden_units x 10 ndarray representing weights between output units and hidden units
    :param hidden_unit_values: num_hidden_unit x 1 ndarray representing the hidden unit values obtained after feed forward
    :param output_unit_values: 10 x 1 ndarray representing the output unit values obtained after feed forward
    """
    assert training_example_label.shape[0] == output_unit_values.shape[0]
    output_unit_error_terms = []
    for idx, output_unit_value in enumerate(output_unit_values):
        output_unit_error = output_unit_value * (1 - output_unit_value) * (
            training_example_label[idx] - output_unit_value)
        output_unit_error_terms.append(output_unit_error)

    # Calculate error for each hidden unit
    hidden_unit_error_terms = []
    for i, hidden_unit_value in enumerate(hidden_unit_values):
        hidden_unit_error_term = 0.0
        for j, output_unit_error_term in enumerate(output_unit_error_terms):
            hidden_unit_error_term += hidden_layer_weights[i, j] * output_unit_error_term
        hidden_unit_error_term = hidden_unit_value * (1 - hidden_unit_value) * hidden_unit_error_term
        hidden_unit_error_terms.append(hidden_unit_error_term)

    return hidden_unit_error_terms, output_unit_error_terms


def update_network_weights(input_layer_weights,
                           hidden_layer_weights,
                           hidden_unit_error_terms,
                           output_unit_error_terms,
                           parameter_factory,
                           training_example,
                           hidden_layer_values):
    """

    :param hidden_layer_values:  num_hidden_units x 1 ndarray representing the values calculated at the hidden layer
    :param training_example: 51 x 1 ndarray representing the input layer values of the training example
    :param input_layer_weights: 51 x num_hidden_units ndarray representing weights between input layer and hidden layer
    :param hidden_layer_weights:  num_hidden_units x 10 ndarray representing weights between hidden layer and output layer
    :param hidden_unit_error_terms: num_hidden_units x 1 ndarray representing error terms of the hidden layer // LIST
    :param output_unit_error_terms: 10 x 1 ndarray representing error terms of the output layer
    :param parameter_factory:
    """
    # Update input_hidden layer weights
    total_input_unit_weight_delta = 0.0
    total_hidden_unit_weight_delta = 0.0
    num_input_units, num_output_units = input_layer_weights.shape
    for i in range(num_input_units):
        for j in range(num_output_units):
            weight_delta = parameter_factory.learning_rate() * hidden_unit_error_terms[j] * \
                           training_example[i]
            input_layer_weights[i, j] += weight_delta
            total_input_unit_weight_delta += weight_delta
    # Update hidden_output layer weights
    num_hidden_units, num_output_units = hidden_layer_weights.shape
    for i in range(num_hidden_units):
        for j in range(num_output_units):
            weight_delta = parameter_factory.learning_rate() * output_unit_error_terms[j] * \
                           hidden_layer_values[i]
            hidden_layer_weights[i, j] += weight_delta
            total_hidden_unit_weight_delta += weight_delta
    print('Weight deltas: Input layer: {} Hidden Layer: {}'.format(total_input_unit_weight_delta,
                                                                   total_hidden_unit_weight_delta))


def do_train(training_data_features, training_data_labels):
    """

    :param training_data_features: 60000 x 50 matrix
    :param training_data_labels:  60000 x 1 matrix
    """
    assert training_data_features.shape[0] == training_data_labels.shape[0]

    parameter_factory = ParameterFactory()
    input_unit_weights, hidden_unit_weights = parameter_factory.initialize_weights()

    # Add bias units to the training features. This makes it 51 units in the input layer with the
    # first input unit being of fixed value 1.0
    training_data_features_with_bias = np.insert(training_data_features, 0, np.full((60000,), 1.0), axis=1)

    for idx, training_example in enumerate(training_data_features_with_bias):
        start_time = time.time()
        hidden_layer_values, output_layer_values = feed_forward(training_example,
                                                                input_unit_weights,
                                                                hidden_unit_weights,
                                                                parameter_factory)
        hidden_unit_error_terms, output_unit_error_terms = back_propagate_errors(training_data_labels[idx],
                                                                                 hidden_unit_weights,
                                                                                 hidden_layer_values,
                                                                                 output_layer_values)

        update_network_weights(input_unit_weights,
                               hidden_unit_weights,
                               hidden_unit_error_terms,
                               output_unit_error_terms,
                               parameter_factory,
                               training_example,
                               hidden_layer_values)
        end_time = time.time()
        print('Training example {} took {} seconds'.format(idx, end_time - start_time))
