import backprop
import data_loader


def initialize_weights():
    pass

# 1 Experiment with bias unit
# 2 Compute Squared loss after every half epoch
# 3 Add momentum
# 4 Plot squared loss
if __name__ == '__main__':
    # Load training data
    input_x_features = data_loader.load_training_data_from_binary()
    input_labels = data_loader.load_training_labels_from_binary()

    # Load testing data
    testing_features = data_loader.load_testing_data_from_binary()
    testing_labels = data_loader.load_testing_labels_from_binary()

    # Train the neural net
    backprop.do_train(input_x_features,
                      data_loader.transform_output_label_into_vector(input_labels),
                      testing_features,
                      data_loader.transform_output_label_into_vector(testing_labels))

    # Test the neural net


