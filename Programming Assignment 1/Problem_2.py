###########################
# Jose Barajas            #
# Majd Abu ghazaleh       #
# Programing Assignment 1 #
#                         #
# June 18, 2019           #
###########################

# Libraries
import numpy as np
from numpy.linalg import inv
from sklearn import datasets
import scipy.io
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# Loading Data

# Diabetes data set
diabetes = datasets.load_diabetes()
x_train_diabetes = diabetes.data[:241]
y_train_diabetes = diabetes.target[:241]

x_test_diabetes = diabetes.data[242:]
y_test_diabetes = diabetes.target[242:]

# Adding in a column of ones for bias/intercept
bias_train_diabetes = np.ones(shape=(len(x_train_diabetes), 1))
bias_test_diabetes = np.ones(shape=(len(x_test_diabetes), 1))
x_train_diabetes_bias = np.append(bias_train_diabetes, x_train_diabetes, axis=1)
x_test_diabetes_bias = np.append(bias_test_diabetes, x_test_diabetes, axis=1)

# Number of features
n_feature_diabetes = x_train_diabetes.shape[1]

#ijcnn
train = scipy.io.loadmat('ijcnn_train.mat')
x_train_ijcnn = train['ijcnn_data'].toarray()
y_train_ijcnn = train['ijcnn_label']

test = scipy.io.loadmat('ijcnn_test.mat')
x_test_ijcnn = test['test_data'].toarray()
y_test_ijcnn = test['test_label']

# Adding in a column of ones for bias/intercept
bias_train_ijcnn = np.ones(shape=(len(x_train_ijcnn), 1))
bias_test_ijcnn = np.ones(shape=(len(x_test_ijcnn), 1))
x_train_ijcnn_bias = np.append(bias_train_ijcnn, x_train_ijcnn, axis=1)
x_test_ijcnn_bias = np.append(bias_test_ijcnn, x_test_ijcnn, axis=1)

# Number of features
n_feature_ijcnn = x_train_ijcnn.shape[1]

#######################################
# Question 2
# Ridge Regression (Diabetes Data Set)
#######################################

def SolverDoDad(weights, X):
    predictions = np.zeros((len(X),1))
    for i in range(0, len(predictions)):
        w_T = np.transpose(weights)
        predictions[i] = np.dot(w_T, X[i])
    return predictions

def RMSE(predictions, actual):
    error = sqrt(mean_squared_error(actual, predictions))
    return error

# Learning Ridge w = (X^TX + N(lambda)I)^-1 * X^T * Y
def RidgeRegressionLearn(X, Y, lam):
    X_T = np.transpose(X)
    # Solving in steps
    # w = (A + B)^-1 * C
    A = np.dot(X_T, X)
    B = np.dot(np.dot(len(X), lam), np.eye(X.shape[1]))
    C = np.dot(X_T, Y)
    w = np.dot(inv(np.add(A, B)), C)
    return w

lambdaSpectrum = [0, 0.001, 0.002, 0.003, 0.004, 0.005, .01, .05, .1, .2, .4, .6, .8, 1.0]
ridgeRSME_train = np.zeros((len(lambdaSpectrum),1))
ridgeRSME_test = np.zeros((len(lambdaSpectrum),1))

for i in range(0,len(lambdaSpectrum)):
    w_Ridge = RidgeRegressionLearn(x_train_diabetes_bias, y_train_diabetes, lambdaSpectrum[i])

    ridgePredictions_train = SolverDoDad(w_Ridge, x_train_diabetes_bias)
    ridgePredictions_test = SolverDoDad(w_Ridge, x_test_diabetes_bias)

    ridgeRSME_train[i] = RMSE(ridgePredictions_train, y_train_diabetes)
    ridgeRSME_test[i] = RMSE(ridgePredictions_test, y_test_diabetes)


# Plotting
plt.figure(100)
plt.plot(lambdaSpectrum, ridgeRSME_test, label='test')
plt.plot(lambdaSpectrum, ridgeRSME_train, label='training')
plt.title("Ridge Regression")
plt.xlabel("Complexity Parameter")
plt.ylabel("RMSE")

plt.legend(title='Data Sets')
plt.show()