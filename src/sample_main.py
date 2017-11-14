import backprop
import data_loader


def initialize_weights():
    pass


if __name__ == '__main__':
    # Load training data
    input_x_features = data_loader.get_sample_data()
    input_labels = data_loader.get_sample_data()

    # Load testing data
    testing_features = data_loader.get_sample_data()
    testing_labels = data_loader.get_sample_data()

    # Train the neural net
    backprop.do_train(input_x_features,
                      input_labels,
                      testing_features,
                      testing_labels)

    # Test the neural net