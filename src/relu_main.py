import relu_backprop
import data_loader
from parameters import HyperParameters

if __name__ == '__main__':
    # Load training data
    input_x_features = data_loader.load_training_data_from_binary()
    input_labels = data_loader.load_training_labels_from_binary()

    # Load testing data
    testing_features = data_loader.load_testing_data_from_binary()
    testing_labels = data_loader.load_testing_labels_from_binary()

    # Train the neural net
    for par in range(1):
        relu_backprop.do_train(input_x_features,
                          data_loader.transform_output_label_into_vector(input_labels),
                          testing_features,
                          data_loader.transform_output_label_into_vector(testing_labels),
                          parameters=HyperParameters(num_hidden_units=500,
                                                     idx=4,
                                                     low=-0.05,
                                                     high=0.05,
                                                     num_input_units=51,
                                                     num_output_units=10,
                                                     momentum=0.0))
