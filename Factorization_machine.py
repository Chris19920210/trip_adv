"""
All codes are written in python 2

Goal:
Choose the hyperparameters

Usage:
In any terminal or shell, cd to working directory and type "python Factorization_machine.py" then the result will be printed in the shell.

Modules:
pandas, sklearn, numpy, fastfm and pickle
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
from fastFM import mcmc

# load the data
data = pickle.load(open('data.pkl', 'rb'))
X = data[0]
y = data[1]
# train_test_split, provide fixed seed for being reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 6)

# build the fm model by mcmc and evaluate the model through test set
def predict_fastfm(trainX, trainY, testX,rank=2, n_iter=100):
    clf = mcmc.FMRegression(rank=rank, n_iter=n_iter)
    return clf.fit_predict(trainX, trainY, testX)

# choose the rank(for further analysis, cross-validation can be used. For saving the time, we use simpler method here)
def main():
    result = {'mse' + str(k) : 0 for k in range(1,9)}
    for i in range(1,9):
        y_pred = predict_fastfm(X_train,y_train,X_test,rank = i)
        result['mse' + str(i)] = mean_squared_error(y_test, y_pred)

    # print the result
    for key, value in result.items():
        print key + ': ' + str(value)

if __name__ == "__main__":
    main()
