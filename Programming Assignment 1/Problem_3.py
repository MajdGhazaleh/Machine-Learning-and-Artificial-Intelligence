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

##########################################################
# Question 3
# Ridge Regression w/ Gradient Descent (Diabetes Data Set)
##########################################################

def SolverDoDad(weights, X):
    predictions = np.zeros((len(X),1))
    for i in range(0, len(predictions)):
        w_T = np.transpose(weights)
        predictions[i] = np.dot(w_T, X[i])
    return predictions

def RMSE(predictions, actual):
    error = sqrt(mean_squared_error(actual, predictions))
    return error


lambdaSpectrum = [0, 0.001, 0.002, 0.003, 0.004, 0.005, .01, .05, .1, .2, .4, .6, .8, 1.0]

def RidgeObjFunction(initialWehights, *args):
    train_data, label, lam = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    train_data = train_data.reshape((n_data,n_features))
    error = np.linalg.norm(np.dot(train_data, initialWehights)-label)**2 + (lam * (np.linalg.norm(initialWehights)**2))
    error_grad = 1/n_data * np.dot(np.transpose(train_data), np.dot(train_data, initialWehights)-label) + (2 * lam * np.linalg.norm(initialWehights))
    error_grad = error_grad.reshape((n_features,))
    return error, error_grad


ridgeRSME_CG_train = np.zeros((len(lambdaSpectrum),1))
ridgeRSME_CG_test = np.zeros((len(lambdaSpectrum),1))
for i in range(0,len(lambdaSpectrum)):
    args_Ridge = (x_train_diabetes_bias, y_train_diabetes, lambdaSpectrum[i])
    ridgeInitialWeights = np.zeros((n_feature_diabetes+1, 1)) # added the plus one to account for bias!
    optsRidge = {'maxiter':100}
    linearParamsRidge = minimize(RidgeObjFunction, ridgeInitialWeights, jac=True, args=args_Ridge, method='CG', options=optsRidge)
    wRidge_CG = linearParamsRidge.x

    ridgeRSME_CG_train[i] = RMSE(SolverDoDad(wRidge_CG, x_train_diabetes_bias), y_train_diabetes)
    ridgeRSME_CG_test[i] = RMSE(SolverDoDad(wRidge_CG, x_test_diabetes_bias), y_test_diabetes)

# Plotting
plt.figure(200)
plt.plot(lambdaSpectrum, ridgeRSME_CG_test, label='test')
plt.plot(lambdaSpectrum, ridgeRSME_CG_train, label='training')
plt.title("Ridge Regression via Gradient Descent")
plt.xlabel("Complexity Parameter")
plt.ylabel("RMSE")

plt.legend(title='Data Sets')
plt.show()

# Comments:
# Overall it appears that the gradient descent method produces similar results to problem 2