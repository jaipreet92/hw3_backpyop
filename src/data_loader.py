import numpy as np


def load_training_data_from_binary():
    raw_file = open('../data/train-images-pca.idx2-double', 'rb')
    raw_file.seek(12)

    # 300000 x 1
    flattened_data = np.fromfile(file=raw_file, dtype=np.double, count=-1)
    # 60000 x 50
    input_x_feature_matrix = np.reshape(flattened_data, (60000, 50), order='C')
    print(input_x_feature_matrix.shape)
    return input_x_feature_matrix

