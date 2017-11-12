import numpy as np


def load_training_data_from_binary():
    raw_file = open('../data/train-images-pca.idx2-double', 'rb')
    raw_file.seek(12)

    # 300000 x 1
    flattened_data = np.fromfile(file=raw_file, dtype=np.dtype('float64').newbyteorder('>'), count=-1)
    # 60000 x 50
    input_x_feature_matrix = np.reshape(flattened_data, (60000, 50), order='C')
    print(input_x_feature_matrix.shape)
    return input_x_feature_matrix


def load_training_labels_from_binary():
    raw_file = open('../data/train-labels.idx1-ubyte', 'rb')
    raw_file.seek(8)
    labels = np.fromfile(file=raw_file, dtype=np.ubyte, count=-1)
    print(labels.shape)
    return labels


def load_testing_data_from_binary():
    raw_file = open('../data/t10k-images-pca.idx2-double', 'rb')
    raw_file.seek(12)

    flattened_data = np.fromfile(file=raw_file, dtype=np.dtype('float64').newbyteorder('>'), count=-1)
    # 10000 x 50
    input_x_feature_matrix = np.reshape(flattened_data, (10000, 50), order='C')
    print(input_x_feature_matrix.shape)
    return input_x_feature_matrix


def load_testing_labels_from_binary():
    raw_file = open('../data/t10k-labels.idx1-ubyte', 'rb')
    raw_file.seek(8)
    labels = np.fromfile(file=raw_file, dtype=np.ubyte, count=-1)
    print(labels.shape)
    return labels


def transform_output_label_into_vector(training_data_labels):
    rows = training_data_labels.shape[0]
    label_vector = np.zeros((rows, 10))

    for i in range(rows):
        idx = training_data_labels[i]  # The digit being represented
        label_vector[i, idx] = 1.0

    return label_vector
