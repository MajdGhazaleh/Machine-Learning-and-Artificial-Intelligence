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
from scipy.optimize import minimize
from sklearn.svm import SVC
import time

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

#########################################
# Question 1
# Linear Regression (Diabetes Data Set)
#########################################

# w = (X^T * X)^-1 * X^T * y
def LinearRegressionLearn(X, Y):
    x_transpose = np.transpose(X)

    # Solving via steps:
    A = np.dot(x_transpose, X)
    #B = inv(A)
    B = np.dot(x_transpose, Y)
    C = np.dot(inv(A), x_transpose)

    w = np.dot(inv(A), B)
    return w

def SolverDoDad(weights, X):
    predictions = np.zeros((len(X),1))
    for i in range(0, len(predictions)):
        w_T = np.transpose(weights)
        predictions[i] = np.dot(w_T, X[i])
    return predictions

def RMSE(predictions, actual):
    error = sqrt(mean_squared_error(actual, predictions))
    return error


w_linear = LinearRegressionLearn(x_train_diabetes, y_train_diabetes)
w_linear_bias = LinearRegressionLearn(x_train_diabetes_bias, y_train_diabetes)

# Predictions
linearPredictions_train = SolverDoDad(w_linear, x_train_diabetes)
linearPredictions_test = SolverDoDad(w_linear, x_test_diabetes)
linearPredictions_train_bias = SolverDoDad(w_linear_bias, x_train_diabetes_bias)
linearPredictions_test_bias = SolverDoDad(w_linear_bias, x_test_diabetes_bias)

# RMSE
linearRMSE_train = RMSE(linearPredictions_train, y_train_diabetes)
linearRMSE_test = RMSE(linearPredictions_test, y_test_diabetes)
linearRMSE_train_bias = RMSE(linearPredictions_train_bias, y_train_diabetes)
linearRMSE_test_bias = RMSE(linearPredictions_test_bias, y_test_diabetes)