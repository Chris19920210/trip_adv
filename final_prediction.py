"""
Goal:
Give the final prediction for each user
Usage:
In any terminal or shell, cd to working directory and type "python final_prediction.py" then the result will be saved in the working directory.

"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pickle
import itertools
from fastFM import mcmc

# read the data
data = pickle.load(open('data.pkl', 'rb'))
X = data[0]
y = data[1]

sparse_predict_data_1 = pickle.load(open('sparse_vector_pred1.pkl', 'rb'))
sparse_predict_data_2 = pickle.load(open('sparse_vector_pred2.pkl', 'rb'))

def predict_fastfm(trainX, trainY, testX,rank=2, n_iter=100):
    clf = mcmc.FMRegression(rank=rank, n_iter=n_iter)
    return clf.fit_predict(trainX, trainY, testX)

def result(dataFrame):
    a = dataFrame.apply(lambda row: abs(row['score'] - 1 ), axis = 1)
    tmp = pd.concat([dataFrame[['user', 'hotel']],a], axis = 1)
    tmp2 = tmp.groupby('user')
    result = []
    for i in range(1,4545):
        z = tmp2.get_group(str(i))
        id = z[0].idxmin()
        record = z.loc[[id]]
        result.append([record.iloc[0].user, record.iloc[0].hotel])
    return pd.DataFrame(result,columns = ['user', 'hotel'])


if __name__ == "__main__":
    ## version 1:  delete the data who have already been looked at
    sparse_1 = predict_fastfm(X, y, sparse_predict_data_1[0])
    sparse_1 = pd.DataFrame(sparse_1, columns = ['score'])
    pred1 = pd.concat([sparse_predict_data_1[1], sparse_1], axis = 1)
    result1 = result(pred1)
    result1.to_csv('result_no_looked_at.txt', index=False, sep = '\t')
    ## version 2: don't delete the data who have already been looked at
    sparse_2 = predict_fastfm(X, y, sparse_predict_data_2[0])
    sparse_2 = pd.DataFrame(sparse_2, columns = ['score'])
    pred2 = pd.concat([sparse_predict_data_2[1], sparse_2], axis = 1)
    result2 = result(pred2)
    result2.to_csv('result_have_looked_at.txt', index=False, sep = '\t')


