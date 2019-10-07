import pandas as pd
import numpy as np

def train_val_split(X_train, Y_train, val_size=10000):
    """
    Plit train set into 2 set: train set and validate set

    Input:
        X_train (60000, nH, nW, 1): training data
        Y_train (60000, 10): training labels
        val_size: size of validate set

    Outputs:
        (X_train, Y_train): tuple of train set
        (X_val, Y_val): tuple of validate set
    """
    m = X_train.shape[0] # get number of examples in training set
    # Split training set into 50000 training examples and 10000 validate exemples
    rand_index = np.random.choice(range(m), val_size, replace = False)
    X_val, Y_val = X_train[rand_index,:,:,:], Y_train[rand_index,:]

    # Update training set
    remaning_index = [i for i in range(m) if i not in rand_index]
    X_train, Y_train = X_train[remaning_index], Y_train[remaning_index]

    return (X_train, Y_train), (X_val, Y_val)

def random_minibatches(X, Y, mini_batch_size = 64, seed = 0):
    '''
    Purpose: Shuffle dataset to generate mini-batches used for each epoches
    Input:
      X: input layer with shape [m, nH0, nW0, nC0]
      Y: label layer with shape [m, 10]
      mini_batch_size: the size of each mini_batche. Default: 64.
      seed: Set seed to get different permutation for each epoch
    Output:
      minibatches: list contains all shuffled mini batches.
    '''
    # Get number of examples
    m = X.shape[0]
    minibatches = []
    
    # Set seed
    np.random.seed(seed)
    
    # Permute index
    rand_index = list(np.random.permutation(m))
    
    # Shuffle dataset
    shuffle_X, shuffle_Y = X[rand_index,:,:,:], Y[rand_index,:]
    
    # Compute number of minibatch
    no_minibatches = int(m/mini_batch_size)

    # Loop over no_minibatch and append each minibatch to list
    for i in range(no_minibatches):
        # Slice each minibatch
        minibatch_X = shuffle_X[i*mini_batch_size : i*mini_batch_size + mini_batch_size,:,:,:]
        minibatch_Y = shuffle_Y[i*mini_batch_size : i*mini_batch_size + mini_batch_size, :]
        # Packing
        minibatch = (minibatch_X, minibatch_Y)
        
        # Append to list
        minibatches.append(minibatch)
        
    # Handle the remaining minibatche if the last minibatch is less than enough minibatch_size
    if m % mini_batch_size != 0:
        # Slice each minibatch
        minibatch_X = shuffle_X[no_minibatches*mini_batch_size:m,:,:,:]
        minibatch_Y = shuffle_Y[no_minibatches*mini_batch_size:m,:]
        # Packing
        minibatch = (minibatch_X, minibatch_Y)
        
        # Append to list
        minibatches.append(minibatch)
        
    return minibatches

# minibatches = random_minibatches(X_train, Y_train, mini_batch_size = 64)

# minibatchX0, minibatchY0 = minibatches[0]
# print(minibatchX0.shape, minibatchY0.shape)

# def predict(X, Y):
#     # Calculate the training accuracy and validating accuracy
#     predict_op = tf.arg_max(Z4, 1)
#     correct_prediction = tf.equal(predict_op, Y)

#     # Calculate the accuracy on training set and validating set
#     # Perfome mean to compute accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
#     accuracy = accuracy.eval({X: X, Y: Y})
    
#     return accuracy

def load_data(train_path, test_path):
    """
    Load Fashion-Mnist dataset which contains 70,000 grayscale images in 10 categories. 
    The images show individual articles of clothing at low resolution (28 by 28 pixels).
    There are 60000 images for training and 10000 images for testing.
    
    Labels range from 0-9 which these correspond to the class of clothing:
    Label	Class
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot

    Input:
        train_path: path name of train set (includes X_train, Y_train)
        test_path: path name of test set (includes X_test, Y_test)

    Output:
        Tuple of (X_train, Y_train): training set
        Tuple of (X_test, T_test): test set
    """
    # Import data
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)

    # Separate x and y in training set
    Y_train = train_set["label"]
    X_train = train_set.drop(labels=["label"], axis=1)

    # Separate x and y in test set
    Y_test = test_set["label"]
    X_test = test_set.drop(labels=["label"], axis=1)

    return (X_train.to_numpy(), Y_train.to_numpy()), (X_test.to_numpy(), Y_test.to_numpy())

def convert_to_one_hot(Y, C):
    """
    Convert Y from shape (m, 1) into (m, C) with each column correspond to the actual digit of example

    Input:
        Y: the groundtruth labels with shape (m, 1)
        C: the number of catogories (class)

    Output:
        Y_one_hot: the one-hot form of Y with shape (m, C)
    """
    Y_one_hot = np.eye(C)[Y.reshape(-1)]
    return Y_one_hot