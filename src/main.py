import backprop
import data_loader
from parameters import HyperParameters


def get_hyper_parameters():
    return [
        HyperParameters(num_hidden_units=500,
                        idx=0,
                        low=-0.1,
                        high=0.1),
        HyperParameters(num_hidden_units=500,
                        idx=1,
                        low=-0.2,
                        high=0.2),
        HyperParameters(num_hidden_units=500,
                        idx=2,
                        low=-0.3,
                        high=0.3),
        HyperParameters(num_hidden_units=500,
                        idx=3,
                        low=-0.4,
                        high=0.4),
        HyperParameters(num_hidden_units=500,
                        idx=4,
                        low=-0.5,
                        high=0.5),
        HyperParameters(num_hidden_units=500,
                        idx=5,
                        low=-0.7,
                        high=0.7),
        HyperParameters(num_hidden_units=500,
                        idx=6,
                        low=-1.0,
                        high=1.0),
        HyperParameters(num_hidden_units=500,
                        idx=7,
                        low=-1.0,
                        high=1.0),
        HyperParameters(num_hidden_units=500,
                        idx=8,
                        low=-1.5,
                        high=1.5),
    ]


if __name__ == '__main__':
    # Load training data
    input_x_features = data_loader.load_training_data_from_binary()
    input_labels = data_loader.load_training_labels_from_binary()

    # Load testing data
    testing_features = data_loader.load_testing_data_from_binary()
    testing_labels = data_loader.load_testing_labels_from_binary()

    # Train the neural net
    for par in get_hyper_parameters():
        print('Training HP {} {}'.format(par.num_hidden_unit(), par.idx()))
        backprop.do_train(input_x_features,
                          data_loader.transform_output_label_into_vector(input_labels),
                          testing_features,
                          data_loader.transform_output_label_into_vector(testing_labels),
                          parameters=par)
