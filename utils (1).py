# Utility functions used for HA4

import numpy as np
import os
import sys
import math


def preprocess(input_folder: str):
    """
    Preprocess the original sign language dataset, and save the altered data to new files
    """
    X_raw = np.load(open(input_folder+ '/X.npy', 'rb'))
    Y_raw = np.load(open(input_folder+ '/y.npy', 'rb'))
    print('X_raw shape: {}'.format(X_raw.shape))
    print('Y_raw shape: {}'.format(Y_raw.shape))

    # Add a channel dimension to X_raw
    X_raw = np.expand_dims(X_raw, axis=3)

    # Examination    
    Y_integer = np.argmax(Y_raw, axis=1).reshape((-1, 1))
    print(Y_integer.shape)
    # print(Y_integer[:10,:])
    classes = np.unique(Y_integer)
    print('All classes:', classes)

    # Randomly sample 80% as training data, and the rest 20% as testing data
    np.random.seed(1)
    train_indices = np.array([])
    for c in classes:
        x_ind = np.where(Y_integer == c)[0]
        np.random.shuffle(x_ind)
        train_indices = np.append(train_indices, x_ind[:int(.8 * len(x_ind))])
        print(x_ind[0])
    print(train_indices.shape)
    train_indices = train_indices.astype(int)
    print(train_indices[:10])

    X_train = X_raw[train_indices, :]
    Y_train = Y_raw[train_indices, :]
    X_test = X_raw[[i for i in range(X_raw.shape[0]) if i not in train_indices], :]
    Y_test = Y_raw[[i for i in range(X_raw.shape[0]) if i not in train_indices], :]
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    # Save to file
    np.save(open('X_train.npy', 'wb'), X_train)
    np.save(open('Y_train.npy', 'wb'), Y_train)
    np.save(open('X_test.npy', 'wb'), X_test)
    np.save(open('Y_test.npy', 'wb'), Y_test)


def load_data():
    data_files = ['X_train.npy', 'Y_train.npy', 'X_test.npy', 'Y_test.npy']
    for df in data_files:
        if not os.path.exists(df):
            sys.stderr.write('Make sure that {} is in the current directory'.format(df))
            sys.stderr.flush()
            sys.exit(1)

    X_train = np.load(open('X_train.npy', 'rb'))
    Y_train = np.load(open('Y_train.npy', 'rb'))
    X_test = np.load(open('X_test.npy', 'rb'))
    Y_test = np.load(open('Y_test.npy', 'rb'))

    return X_train, Y_train, X_test, Y_test


def generate_minibatch(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (num_examples, input size)
    Y -- true "label" vector of shape (num_examples, num_classes)
    seed -- this is only for the purpose of grading 
    
    Returns:
    mini_batches -- list of (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    preprocess('../ha3/Sign-language-digits-dataset 2')
    return

if __name__ == '__main__':
    main()