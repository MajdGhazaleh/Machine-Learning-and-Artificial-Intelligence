import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle
import re

#Majd Abu Ghazaleh : majdhish@buffalo.edu
#Jose Barjas : josebara@buffalo.edu

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    z = np.asarray(z)

    return 1/(1 + np.exp(-z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    # Initializing dummies
    index_train_dummy = 0
    index_test_dummy = 0
    train_dummy = np.zeros(shape=(60000, 784))
    test_dummy = np.zeros(shape=(10000, 784))
    train_label_dummy = np.zeros(shape=(60000,))
    test_label_dummy = np.zeros(shape=(10000,))

    # so now I need to go through the train data and make each observation
    # a row in the overall matrix
    # and for training data, need to go through and make each observation
    # grab the relevant ground truth label from the
    # corresponding position in big matrix

    for i in range(10):
        train = mat["train" + str(i)]
        test = mat["test" + str(i)]
        label = i
        index_train = index_train_dummy + train.shape[0]
        index_test = index_test_dummy + test.shape[0]
        train_dummy[index_train_dummy: index_train] = train
        train_label_dummy[index_train_dummy: index_train] = label
        test_dummy[index_test_dummy: index_test] = test
        test_label_dummy[index_test_dummy: index_test] = label
        index_train_dummy = index_train
        index_test_dummy = index_test

    randomSample = np.random.RandomState(seed=54321).permutation(60000)

    train_data = train_dummy[randomSample[0:50000]]
    test_data = test_dummy
    validation_data = train_dummy[randomSample[50000:60000]]

    train_label = train_label_dummy[randomSample[0:50000]]
    test_label = test_label_dummy
    validation_label = train_label_dummy[randomSample[50000:60000]]

    # Normalize the train and validation data
    normalizing_factor = 255
    train_data = train_data / normalizing_factor
    validation_data = validation_data / normalizing_factor

    ##################################################################
    # Feature selection
    # Your code here.

    # then need to do feature selection by choosing the features
    # that are the same across all observations
    # so just need to ignore all features for which values are exactly
    # the same across all data

    duplicates = np.all(train_data == train_data[0, :], axis=0)
    index = np.where(duplicates)[0]
    train_data = np.delete(train_data, index, 1)
    test_data = np.delete(test_data, index, 1)
    validation_data = np.delete(validation_data, index, 1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

    


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    #Your code here
    #fist layer to the second layer the feedforward propagation
    #define the error function wothout regulation


    #calculate the error value without regulation


    #calculate the regulation function


    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    indices = np.arange(training_label.shape[0], dtype="int")
    dummy_Label = np.zeros((training_label.shape[0], 10))
    dummy_Label[indices, training_label.astype(int)] = 1
    training_label = dummy_Label

    size = training_data.shape[0]

    # Input Layer
    input_layer_bias = np.ones((size, 1), dtype=np.int)
    training_data = np.hstack((training_data, input_layer_bias))

    # Hidden Layer
    hidden_layer = sigmoid(np.dot(training_data, w1.T))

    hidden__layer_bias = np.ones((hidden_layer.shape[0], 1))
    hidden_layer = np.hstack((hidden_layer, hidden__layer_bias))

    # Output Layer
    output = sigmoid(np.dot(hidden_layer, w2.T))

    # Back Propogation
    # Applying equations from assignment document

    w1_gradient = np.dot(((1 - hidden_layer) *
                          hidden_layer *
                          (np.dot((output - training_label), w2))).T, training_data)

    w1_gradient = np.delete(w1_gradient, n_hidden, 0)

    w2_gradient = np.dot((output - training_label).T, hidden_layer)

    # adding in both gradients for obj
    obj_grad = np.array([])
    obj_grad = np.concatenate((w1_gradient.flatten(), w2_gradient.flatten()), 0) / size

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    # Calculating Objective Scalar value:

    # Calculating Log-likelihood
    LL = (np.sum(-1 * (training_label * np.log(output) + (1 - training_label) * np.log(1 - output)))) / size
    # Calculating Regularization parameter
    regParameter = (lambdaval / (2 * size)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    # Objective function is made by adding in the log likelyhood and the regularization.
    obj_val = LL + regParameter
    return (obj_val, obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 

    #Your code here

    # Input Layer
    input_layer_bias = np.ones((data.shape[0], 1), dtype=np.int)
    data = np.hstack((data, input_layer_bias))

    # Hidden Layer
    hidden_layer = sigmoid(np.dot(data, w1.T))
    hidden_layer_bias = np.ones((hidden_layer.shape[0], 1))  # Bias addition
    hidden_layer = np.hstack((hidden_layer, hidden_layer_bias))

    # Output Layer =
    output = sigmoid(np.dot(hidden_layer, w2.T))

    # Predictions
    labels = np.argmax(output, axis=1)


    return labels

"""**************Neural Network Script Starts here********************************"""
start=time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

##transfer the y to a 10-dimension vector
#def vectorized_result(y):
#    e=np.zeros(10)
#    e[y]=1.0
#    return e
#y_tdvec=np.array([vectorized_result(y) for y in train_label])

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 35;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);


# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.3;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}    # Preferred value.



nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)



#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

end=time.time()
print(end-start)

final_parameter = (n_hidden, w1, w2, lambdaval)
pickle.dump(final_parameter, open('params.pickle', 'wb'))
