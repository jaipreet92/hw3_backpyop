import numpy as np
import data_loader

if __name__ == '__main__':
    input_x_features = data_loader.load_training_data_from_binary()
    input_labels = data_loader.load_training_labels_from_binary()

    testing_features = data_loader.load_testing_data_from_binary()
    testing_labels = data_loader.load_testing_labels_from_binary()
    print('Placeholder')