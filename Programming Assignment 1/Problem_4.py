###########################
# Jose Barajas            #
# Majd Abu ghazaleh       #
# Programing Assignment 1 #
#                         #
# June 18, 2019           #
###########################

# Libraries
from numpy import *
from pylab import *
import numpy as np
from sklearn import datasets
from scipy.optimize import minimize
import scipy.io
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
# Question 4
# Logistic Regression w/ Gradient Descent (ijcnn Data Set)
##########################################################

n_feature = x_train_diabetes.shape[1]




def sigmoid(X):
    N = 1.0
    D = 1.0 + exp(X * (-1))
    return N / D
    show()


def grad(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * X.T.dot(h - y)
    return (grad.flatten())


def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (log(h).T.dot(y) + log(1 - h).T.dot(1 - y))
    if isnan(J[0]):
        return (inf)
    return J[0]


def predict(theta, X):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    p = np.zeros([m, 1])
    for i in range(m):
        if h[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p, h


X = x_train_ijcnn
y = y_train_ijcnn
init_theta = np.zeros([X.shape[1], 1])
init_cost = costFunction(init_theta, X, y)
res = minimize(costFunction, init_theta, args=(X, y), jac=grad, options={'maxiter': 100})
theta = res.x
p, h = predict(theta, X)


# Fail safe method (accuracy = 92 percent)
y = y_train_ijcnn.ravel()
clf = LogisticRegression().fit(x_train_ijcnn, y)
accuracy = clf.score(x_test_ijcnn, y_test_ijcnn)
