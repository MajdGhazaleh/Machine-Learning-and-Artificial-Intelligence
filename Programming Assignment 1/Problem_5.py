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

#######################
# Question 5
# SVM (ijcnn Data Set
######################
def SupportVectorize(X_train, X_test,  Y_train, kern, gamm='auto', c=1.0):
    print("Learning via", kern,"kernel...")
    t0 = time.time()
    svmKernel = SVC(C=c, kernel=kern, gamma=gamm)
    svmKernel.fit(X_train, np.ravel(Y_train))
    t1 = time.time()
    t = (t1 - t0) / 60
    print("done learning after", t, "minutes")
    print("Predicting...")
    svmPredict_test = svmKernel.predict(X_test)
    svmPredict_train = svmKernel.predict(X_train)
    print("Done Predicting")
    return svmPredict_test, svmPredict_train


def Accuracy(predictions, actual):
    actual = np.ravel(actual)
    # Initializing
    correct = 0
    wrong = 0
    for k in range(0,len(actual)):
        if predictions[k] == actual[k]:
            correct = correct + 1
        else:
            wrong = wrong + 1
    accuracy = (correct/len(actual)) * 100
    return accuracy

# Linear
svmLinearPredict_test, svmLinearPredict_train = SupportVectorize(x_train_ijcnn, x_test_ijcnn, y_train_ijcnn, kern='linear')

svmLinearAccuracy_test = Accuracy(svmLinearPredict_test, y_test_ijcnn)
svmLinearAccuracy_train = Accuracy(svmLinearPredict_train, y_train_ijcnn)

# RBF, Gamma = 1
svmRBFPredict_test, svmRBFPredict_train = SupportVectorize(x_train_ijcnn, x_test_ijcnn, y_train_ijcnn, kern='rbf', gamm=1.0)

svmRBFAccuracy_test = Accuracy(svmRBFPredict_test, y_test_ijcnn)
svmRBFAccuracy_train = Accuracy(svmRBFPredict_train, y_train_ijcnn)

# RBF, Gamma = default
svmRBFPredict_test_d, svmRBFPredict_train_d = SupportVectorize(x_train_ijcnn, x_test_ijcnn, y_train_ijcnn, kern='rbf')

svmRBFAccuracy_test_d = Accuracy(svmRBFPredict_test_d, y_test_ijcnn)
svmRBFAccuracy_train_d = Accuracy(svmRBFPredict_train_d, y_train_ijcnn)

# RBF, Gamma = default, C = 1, 10, 20, ...100)
cSpectrum = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

svmRBFAccuracy_d_c_test = np.zeros((len(cSpectrum), 1))
svmRBFAccuracy_d_c_train = np.zeros((len(cSpectrum), 1))
print("Alright, give me a moment")
t0 = time.time()
for k in range(0, len(cSpectrum)-1):
    predictions_test, predictions_train = SupportVectorize(x_train_ijcnn, x_test_ijcnn, y_train_ijcnn, kern='rbf', c=cSpectrum[k])
    svmRBFAccuracy_d_c_test[k] = Accuracy(predictions_test, y_test_ijcnn)
    svmRBFAccuracy_d_c_train[k] = Accuracy(predictions_train, y_train_ijcnn)
t1 = time.time()
t = (t1 - t0)/60
print("Done computing everything after", t, "minutes")

# Plotting

plt.figure(500)

plt.plot(cSpectrum, svmRBFAccuracy_d_c_test, label='test')
plt.plot(cSpectrum, svmRBFAccuracy_d_c_train, label='training')
plt.title("SVM, Radial Basis Function, Default Gamma, Varying C")
plt.xlabel("Complexity Parameter, C")
plt.ylabel("Accuracy in %")

plt.legend(title='Data Sets')
plt.show()

bestCValue_test = cSpectrum[np.where(svmRBFAccuracy_d_c_test == max(svmRBFAccuracy_d_c_test))[0][0]]
bestCValue_train = cSpectrum[np.where(svmRBFAccuracy_d_c_train == max(svmRBFAccuracy_d_c_train))[0][0]]