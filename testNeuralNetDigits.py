"""
======================================================
Test of Neural Nets on Digits data
======================================================

Author: Michael O'Meara, 2014

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

from nn import NeuralNet

# load the data set

filename = 'data/digitsX.dat'
file = open(filename, 'r')
digitsX = np.loadtxt(file, delimiter=',')

filename = 'data/digitsY.dat'
file = open(filename, 'r')
digitsY = np.loadtxt(file, delimiter=',')
# dataset = datasets.load_digits()

X = digitsX[:]
y = digitsY[:]

n,d = X.shape
nTrain = 0.2*n  #training on 50% of the data

# shuffle the data
# idx = np.arange(n)
# np.random.seed(13)
# np.random.shuffle(idx)
# X = X[idx]
# y = y[idx]

# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
# Xtest = X[nTrain:,:]
# ytest = y[nTrain:]

model = NeuralNet(np.array([25]), .80, 0.12, 600)  # 100 @ 2.5 = 0.885, 400 @ 1.6 = 0.88, 1000 @ 1 = 0.8542, 
model.fit(X,y)
ypred = model.predict(Xtrain)

accuracy = accuracy_score(ytrain, ypred)

print "NeuralNet Accuracy = "+str(accuracy)

# model.visualizeHiddenNodes('hiddenLayers.png')
