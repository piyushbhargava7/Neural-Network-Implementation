import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed

seed(100)

def tot_error(pred, actual):
    '''
    calculating sum-of-squared errors
    :param pred: predicted y
    :param actual: actual y
    :return: total error
    '''
    temp = pred - actual
    error = 0
    for i in temp:
        error+= i*i
    return error


def sigmoid(x):
    '''
    calculating sigmoid of given input
    :param x: input
    :return: sigmoid of input
    '''
    return 1 / (1 + math.exp(-x))


def sqr(x):
    '''
    calculating square of x
    :param x: x
    :return: square of x
    '''
    return x*x


def sample(obs):
    '''
    creating the sample of observations (X,Y) based on the model in (b)
    :param obs: number of obs in sample
    :return: X,Y of the sample
    '''
    a1 = np.array([3,3])
    a2 = np.array([3,-3])
    X= np.random.randn(obs,2)
    Z = np.random.randn(obs)
    Y = np.array([[sigmoid(np.dot(X[j],a1)) + sqr(np.dot(X[j],a2)) + 0.30*Z[j]] for j in range(obs)])
    return X, Y


# Part 2(a)
# Fitting neural network model
def neural_network(learning_rate, weight_decay, hidden_dim, n_iterations, x_train, y_train):
    '''
    fits the neural network and returns the weight parameters of fit model
    :param learning_rate: learning rate
    :param weight_decay: regularization parameter
    :param hidden_dim: Number of hidden units
    :param n_iterations: number of iterations
    :param x_train: train data features
    :param y_train: train data output
    :return:
    '''
    # intializing weights of layer 1
    weights_1 = np.random.random((2,hidden_dim))
    # intializing b1(biases) of layer 1
    b1 = np.random.random((hidden_dim,1))
    # intializing weights of layer 2
    weights_2 = np.random.random((hidden_dim,1))
    # intializing b2(bias) of layer 2
    b2 = np.random.random()

    test_error = []
    train_error = []

    for j in xrange(n_iterations):
        # Computing layer_1 and layer_2 (y) outputs using weights
        layer_1 = 1/(1+np.exp(-(np.dot(x_train,weights_1)+b1.T)))
        layer_2 = np.dot(layer_1,weights_2) + b2

        # Backpropagation - layer 2
        layer_2_delta = 2*(layer_2 - y_train)
        dw2 = (layer_1.T).dot(layer_2_delta)
        db2 = np.sum(layer_2_delta, axis=0, keepdims=True)

        # Backpropagation - layer 1
        layer_1_delta = layer_2_delta.dot(weights_2.T) * (layer_1 * (1-layer_1))
        dw1 = x_train.T.dot(layer_1_delta)
        db1 = np.array([np.sum(layer_1_delta, axis=0)]).T

        # Adding regularization terms to weights
        dw2 += weight_decay * weights_2
        dw1 += weight_decay * weights_1

        # Updating parameters using gradient descent
        weights_2 += -learning_rate *dw2
        b2 += -learning_rate *db2
        weights_1 += -learning_rate *dw1
        b1 += -learning_rate *db1


    return weights_2, b2, weights_1, b1


# Creating training and test samples
x_train, y_train = sample(100)
x_test, y_test = sample(1000)


# Fitting a model on training and evaluating error on test data
# Training
weights_2, b2, weights_1, b1=neural_network(0.0005, 0.01, 10, 10000, x_train, y_train)

# Evaluating on test
layer_1_test = 1/(1+np.exp(-(np.dot(x_test,weights_1)+b1.T)))
layer_2_test = np.dot(layer_1_test,weights_2) + b2
print "Error using the Neural network on a test sample of 1000 observations (after 10,000 iterations)" \
      " is %.2f" %tot_error(layer_2_test,y_test)[0]


# Part 2(b)
# Plotting training and test error curves for different values of weight decay
def neural_network_training_test_error(learning_rate, weight_decay, hidden_dim, n_iterations,
                                       x_train, y_train, x_test, y_test):
    # intializing weights of layer 1
    weights_1 = np.random.random((2,hidden_dim))
    # intializing b1(biases) of layer 1
    b1 = np.random.random((hidden_dim,1))
    # intializing weights of layer 2
    weights_2 = np.random.random((hidden_dim,1))
    # intializing b2(bias) of layer 2
    b2 = np.random.random()

    test_error = []
    train_error = []

    for j in xrange(n_iterations):
        # Computing layer_1 and layer_2 (y) outputs using weights
        layer_1 = 1/(1+np.exp(-(np.dot(x_train,weights_1)+b1.T)))
        layer_2 = np.dot(layer_1,weights_2) + b2

        # Backpropagation - layer 2
        layer_2_delta = 2*(layer_2 - y_train)
        dw2 = (layer_1.T).dot(layer_2_delta)
        db2 = np.sum(layer_2_delta, axis=0, keepdims=True)

        # Backpropagation - layer 1
        layer_1_delta = layer_2_delta.dot(weights_2.T) * (layer_1 * (1-layer_1))
        dw1 = x_train.T.dot(layer_1_delta)
        db1 = np.array([np.sum(layer_1_delta, axis=0)]).T

        # Adding regularization terms to weights
        dw2 += weight_decay * weights_2
        dw1 += weight_decay * weights_1

        # Updating parameters using gradient descent
        weights_2 += -learning_rate *dw2
        b2 += -learning_rate *db2
        weights_1 += -learning_rate *dw1
        b1 += -learning_rate *db1

        # Computing train and test error at end of iteration
        train_error.append(tot_error(layer_2,y_train))

        layer_1_test = 1/(1+np.exp(-(np.dot(x_test,weights_1)+b1.T)))
        layer_2_test = np.dot(layer_1_test,weights_2) + b2
        test_error.append(tot_error(layer_2_test,y_test))


    return train_error, test_error



# Weight decay = 0.01
train_error_a, test_error_a = neural_network_training_test_error(0.0005, 0.01, 10, 10000,
                                       x_train, y_train, x_test, y_test)


# Weight decay = 0.1
train_error_b, test_error_b = neural_network_training_test_error(0.0005, 0.1, 10, 10000,
                                       x_train, y_train, x_test, y_test)


# Weight decay = 1
train_error_c, test_error_c = neural_network_training_test_error(0.0005, 1, 10, 10000,
                                       x_train, y_train, x_test, y_test)



# Weight decay = 2
train_error_d, test_error_d = neural_network_training_test_error(0.0005, 2, 10, 10000,
                                       x_train, y_train, x_test, y_test)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(10000), train_error_a, linestyle='--',color='r', label='Train,  lambda = 0.01')
plt.plot(np.arange(10000), test_error_a, color='r', label='Test , lambda = 0.01')
plt.plot(np.arange(10000), train_error_b, linestyle='--', color='b', label='Train,  lambda = 0.1')
plt.plot(np.arange(10000), test_error_b, color='b', label='Test , lambda = 0.1')
plt.plot(np.arange(10000), train_error_c, linestyle='--', color='g', label='Train,  lambda = 1')
plt.plot(np.arange(10000), test_error_c, color='g', label='Test , lambda = 1')
plt.plot(np.arange(10000), train_error_d, linestyle='--', color='y', label='Train,  lambda = 2')
plt.plot(np.arange(10000), test_error_d, color='y', label='Test , lambda = 2')
plt.title("Train and Test error for different weight decay / lambda using NN")
plt.xlabel("Number of Training Epochs")
plt.ylabel("Sum of squares Error")
plt.legend(loc='upper right', fontsize=12)
plt.grid()
plt.savefig("2_a.png", format="png")
plt.show()




# Part 2(c)
# Varying number of hidden units - Using weight decay as 0.01 which results in least test error (from 2 (b))

test_error_hidden_units=[]

for i in range(10):
    train_error, test_error = neural_network_training_test_error(0.0005, 0.01, i+1, 10000,
                                           x_train, y_train, x_test, y_test)

    test_error_hidden_units.append(test_error[-1])
    print "The test error for hidden units %f" %(i+1), "is %0.2f" %test_error[-1][0]


# The test error for hidden units 1.000000 is 698522.29
# The test error for hidden units 2.000000 is 408437.52
# The test error for hidden units 3.000000 is 93189.87
# The test error for hidden units 4.000000 is 92385.75
# The test error for hidden units 5.000000 is 96093.94
# The test error for hidden units 6.000000 is 29721.26
# The test error for hidden units 7.000000 is 29191.28
# The test error for hidden units 8.000000 is 26893.63
# The test error for hidden units 9.000000 is 25521.79
# The test error for hidden units 10.000000 is 20546.24

