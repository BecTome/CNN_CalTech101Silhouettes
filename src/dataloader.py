import src.config as config
import os
import numpy as np
import scipy.io as io
from sklearn.model_selection import train_test_split

def read_train(path,):
    train_data = io.loadmat(os.path.join(path, "train", 'train_data.mat'))
    X_train, y_train = train_data['X'], train_data['y']
    y_train = y_train - np.min(y_train)
    return X_train, y_train

def read_val(path):
    val_data = io.loadmat(os.path.join(path, "val", 'val_data.mat'))
    X_val, y_val = val_data['X'], val_data['y']
    y_val = y_val - np.min(y_val)
    return X_val, y_val

def read_test(path):
    test_data = io.loadmat(os.path.join(path, "test", 'test_data.mat'))
    X_test, y_test = test_data['X'], test_data['y']
    y_test = y_test - np.min(y_test)
    return X_test, y_test

def create_partition(input_path, val_ratio, test_ratio, output_path, seed=42):

    data = io.loadmat(input_path)
    X_train, y_train = data['train_data'], data['train_labels']

    X_val, y_val = data['val_data'], data['val_labels']

    X_test, y_test = data['test_data'], data['test_labels']

    X = np.concatenate((X_train, X_val, X_test))
    y = np.concatenate((y_train, y_val, y_test))

    N = X.shape[0]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y,
                                                            test_size=int(test_ratio * N),
                                                            random_state=seed,
                                                            stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                        test_size=int(val_ratio * N),
                                                        random_state=seed,
                                                        stratify=y_trainval)
    
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == N

    # Reshape them as images
    X_train = X_train.reshape((-1, config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS))
    X_val = X_val.reshape((-1, config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS))
    X_test = X_test.reshape((-1, config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS))

    # Create directories for train, val y test
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    test_path = os.path.join(output_path, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Save images
    io.savemat(os.path.join(train_path, 'train_data.mat'), {'X': X_train, 'y': y_train})
    io.savemat(os.path.join(val_path, 'val_data.mat'), {'X': X_val, 'y': y_val})
    io.savemat(os.path.join(test_path, 'test_data.mat'), {'X': X_test, 'y': y_test})
