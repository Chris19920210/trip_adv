"""
All codes are written in python 2

Goals:
1. Read the data by pandas
2. Merge the data
3. Change the Dataframes to dictionary
4. Vectorize the dictionary by scikit-learn for further analysis

Usage:
In any terminal or shell, cd to working directory and type "python feature_engineering.py" then the all kinds of dataset objects will appear in working directory

Modules:
pandas, sklearn, numpy and pickle
"""


import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pickle
import itertools

#read data (train)
user_data = pd.read_table('./users.txt',dtype={'user':object, 'home continent': object})
activity_data = pd.read_table('./activity.txt',dtype={'user':object, 'hotel':object})
hotels_data = pd.read_table('./hotels.txt',dtype={'hotel':object})

# predictt set(predict)
user_column =  pd.DataFrame(list(itertools.chain.from_iterable([[str(i)]*66 for i in range(1,4545)])),columns=['user'])
hotel_column = pd.DataFrame([str(i) for i in range(1,67)]*4544,columns=['hotel'])
full_stack = pd.concat([user_column,hotel_column],axis = 1)

def all_train_data():
    # merge data
    user_activity = pd.merge(user_data, activity_data, on = 'user')
    final_data = pd.merge(user_activity,hotels_data, on = 'hotel')

    # change to dict
    final_dict = final_data.T.to_dict().values()

    # Dictionary to vector
    v = DictVectorizer()
    X = v.fit_transform(final_dict)
    y = np.repeat(1.0,X.shape[0])

    return (X, y)

## prediction feature engineering
## version 1:  delete the data who have already been looked at
def sparse_vector_pred_1():
    full_stack.set_index(['user', 'hotel'], inplace=True)
    activity_data.set_index(['user', 'hotel'], inplace=True)
    predict_data_1 = full_stack[~full_stack.index.isin(activity_data.index)].reset_index()
    predict_data_1 = pd.merge(predict_data_1, user_data, on = 'user')
    predict_data_1 = pd.merge(predict_data_1, hotels_data, on = 'hotel')
    dict_predict_data_1 = predict_data_1.T.to_dict().values()
    v = DictVectorizer()
    sparse_predict_data_1 = v.fit_transform(dict_predict_data_1)
    return (sparse_predict_data_1, predict_data_1[['user', 'hotel']])

def sparse_vector_pred_2():
    predict_data_2 = pd.merge(full_stack, user_data, on = 'user')
    predict_data_2 = pd.merge(predict_data_2, hotels_data, on = 'hotel')
    dict_predict_data_2 = predict_data_2.T.to_dict().values()
    v = DictVectorizer()
    sparse_predict_data_2 = v.fit_transform(dict_predict_data_2)
    return (sparse_predict_data_2, predict_data_2[['user','hotel']])


if __name__ == "__main__":
    # save the features as pickle object for further analysis
    data = all_train_data()
    output = open('data.pkl','wb')
    pickle.dump(data, output)
    sparse_vector_pred_2 = sparse_vector_pred_2()
    output = open('sparse_vector_pred2.pkl','wb')
    pickle.dump(sparse_vector_pred_2, output)
    sparse_vector_pred_1 = sparse_vector_pred_1()
    output = open('sparse_vector_pred1.pkl','wb')
    pickle.dump(sparse_vector_pred_1, output)



