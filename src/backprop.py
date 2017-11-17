import numpy as np
import scipy.special as special
import time
from sklearn.preprocessing import scale

from parameters import HyperParameters


def feed_forward(training_example,
                 input_layer_weights,
                 hidden_layer_weights):
    """
    Feed forward part of the the BackPropagation algorithmn
    :rtype: object
    :param training_example: 51 x 1 ndarray
    :param input_layer_weights: 51 x num_hidden_units ndarray
    :param hidden_layer_weights: num_hidden_units x 10 ndarray
    """
    # calculate hidden layer values
    hidden_layer_values = special.expit(np.dot(training_example, input_layer_weights))
    # calculate output layer values
    output_layer_values = special.expit(np.dot(hidden_layer_values, hidden_layer_weights))
    return hidden_layer_values, output_layer_values


def back_propagate_errors(training_example_label,
                          hidden_layer_weights,
                          hidden_unit_values,
                          output_unit_values):
    """

    :param training_example_label: 10 x 1 ndarray for actual output layer or label for current training example
    :param hidden_layer_weights: num_hidden_units x 10 ndarray representing weights between output units and hidden units
    :param hidden_unit_values: num_hidden_unit x 1 ndarray representing the hidden unit values obtained after feed forward
    :param output_unit_values: 10 x 1 ndarray representing the output unit values obtained after feed forward
    """
    assert training_example_label.shape[0] == output_unit_values.shape[0]

    output_unit_error_terms = output_unit_values * (np.full(output_unit_values.shape, 1.0) - output_unit_values) * (
        training_example_label - output_unit_values)

    hidden_unit_error_terms = np.dot(hidden_layer_weights, output_unit_error_terms) * hidden_unit_values * (
        np.full(hidden_unit_values.shape, 1.0) - hidden_unit_values)

    return hidden_unit_error_terms, output_unit_error_terms


def get_weight_delta(hidden_unit_error_terms,
                     output_unit_error_terms,
                     parameter_factory,
                     training_example,
                     hidden_layer_values):
    """

    :param hidden_layer_values:  num_hidden_units x 1 ndarray representing the values calculated at the hidden layer
    :param training_example: 51 x 1 ndarray representing the input layer values of the training example
    :param hidden_unit_error_terms: num_hidden_units x 1 ndarray representing error terms of the hidden layer // LIST
    :param output_unit_error_terms: 10 x 1 ndarray representing error terms of the output layer
    :param parameter_factory:
    """
    input_unit_weights_delta = np.outer(training_example, hidden_unit_error_terms) * parameter_factory.learning_rate()
    hidden_unit_weights_delta = np.outer(hidden_layer_values,
                                         output_unit_error_terms) * parameter_factory.learning_rate()
    return input_unit_weights_delta, hidden_unit_weights_delta


def do_train(training_data_features,
             training_data_labels,
             testing_data_features,
             testing_data_labels):
    """

    :param training_data_features: 60000 x 50 matrix
    :param training_data_labels:  60000 x 1 matrix
    """
    assert training_data_features.shape[0] == training_data_labels.shape[0]

    # Replace nan values in the training data
    replace_nan_values(training_data_features, training_data_labels)

    # Initialize training parameters
    parameters = HyperParameters(num_hidden_units=100,
                                 num_epochs=30,
                                 num_input_units=51,
                                 num_output_units=10,
                                 mini_batch_size=1,
                                 learning_rate=0.1)
    input_unit_weights, hidden_unit_weights = parameters.initialize_weights()

    training_data_features = scale(training_data_features, axis=0, with_mean=True, with_std=True)
    testing_data_features = scale(testing_data_features, axis=0, with_mean=True, with_std=True)

    # Add bias units to the features.
    training_data_features = np.insert(training_data_features, 0,
                                       np.full((training_data_features.shape[0],), 1.0), axis=1)
    testing_data_features = np.insert(testing_data_features, 0,
                                       np.full((testing_data_features.shape[0],), 1.0), axis=1)

    input_unit_weights_delta = np.zeros(input_unit_weights.shape)
    hidden_unit_weights_delta = np.zeros(hidden_unit_weights.shape)
    for n in range(parameters.num_epochs()):
        for idx, training_example in enumerate(training_data_features):
            # SGD
            if idx % parameters.mini_batch_size() == 0:
                input_unit_weights = input_unit_weights + input_unit_weights_delta
                hidden_unit_weights = hidden_unit_weights + hidden_unit_weights_delta
                input_unit_weights_delta = np.zeros(input_unit_weights.shape)
                hidden_unit_weights_delta = np.zeros(hidden_unit_weights.shape)

            hidden_layer_values, output_layer_values = feed_forward(training_example,
                                                                    input_unit_weights,
                                                                    hidden_unit_weights)
            hidden_unit_error_terms, output_unit_error_terms = back_propagate_errors(training_data_labels[idx],
                                                                                     hidden_unit_weights,
                                                                                     hidden_layer_values,
                                                                                     output_layer_values)
            curr_input_unit_weights_delta, curr_hidden_unit_weights_delta = get_weight_delta(hidden_unit_error_terms,
                                                                                             output_unit_error_terms,
                                                                                             parameters,
                                                                                             training_example,
                                                                                             hidden_layer_values)
            input_unit_weights_delta += curr_input_unit_weights_delta
            hidden_unit_weights_delta += curr_hidden_unit_weights_delta
        if n % 1 == 0:
            total_square_error, correct_predictions, incorrect_predictions = get_squared_error(testing_data_features,
                                                                                               testing_data_labels,
                                                                                               input_unit_weights,
                                                                                               hidden_unit_weights)
            print('n {} : SE: {} Correct: {} Incorrect: {}'.format(n, total_square_error, correct_predictions,
                                                                   incorrect_predictions))
    print('Wait!')


def get_squared_error(testing_data_features, testing_data_labels, input_unit_weights, hidden_unit_weights):
    total_square_error = 0.0
    correct_predictions = 0.0
    incorrect_predictions = 0.0
    for idx, test_example in enumerate(testing_data_features):
        predicted_values = feed_forward(test_example, input_unit_weights, hidden_unit_weights)[1]
        actual_values = testing_data_labels[idx]
        total_square_error += np.sum(np.square(predicted_values - actual_values))

        if np.argmax(testing_data_labels[idx]) == np.argmax(predicted_values):
            correct_predictions += 1.0
        else:
            incorrect_predictions += 1.0

    if total_square_error == 0.0:
        raise ValueError('0 error ?!')
    return total_square_error / (2.0 * testing_data_labels.shape[0]), correct_predictions, incorrect_predictions


def replace_nan_values(training_data, training_data_labels):
    """
    Replaces 'nan' values in the training data with 0.0, as these cause problems down the line when using these
    values
    :param training_data:
    :param training_data_labels:
    """
    if np.any(np.isnan(training_data)):
        np.nan_to_num(training_data, copy=False)
    if np.any(np.isnan(training_data_labels)):
        np.nan_to_num(training_data_labels, copy=False)
