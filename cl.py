
"""
Goal:
build the recommender system based on collaborative filtering.
This method is a variant of CL, the recommender system is not only based on rating matrix but also based on user features and hotel feature.

Usage:
In any terminal or shell, cd to working directory and type "python cl.py" then the result will be saved in the working directory.
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pickle
import itertools
from collections import defaultdict
#read data (train)
user_data = pd.read_table('./users.txt',dtype={'user':object, 'home continent': object})
activity_data = pd.read_table('./activity.txt',dtype={'user':object, 'hotel':object})
hotels_data = pd.read_table('./hotels.txt',dtype={'hotel':object})
hotels_dict = hotels_data.set_index('hotel').to_dict().values()[0]
# merge the data
user_activity = pd.merge(user_data, activity_data, on = 'user')
final_data = pd.merge(user_activity,hotels_data, on = 'hotel')

# the tuple for look_up_dictionary construction
tmp = zip(final_data['user'], final_data['hotel'], final_data['star_rating'])
# look up dictionay
Look_up_dict = defaultdict(dict)
for i in tmp:
    try:
        Look_up_dict[i[0]][i[1]] = i[2]
    except:
        Look_up_dict[i[0]] = {i[1] : i[2]}

# user_similarity weights(very time consuming)
def user_similarity(user_data):
    user_list = zip(user_data['user'],user_data['home continent'], user_data['gender'])
    dict1 = defaultdict(list)
    for i in user_list:
        dict1[i[0]] = [i[1],i[2]]
    dict2 = defaultdict(float)
    for key,value in dict1.items():
        for key2, value2 in dict1.items():
            if len(set(value) - set(value2)) == 0:
                dict2[(key, key2)] = 1.4
            elif len(set(value) - set(value2)) == 1:
                dict2[(key, key2)] = 1.2
            else:
                dict2[(key, key2)] = 1
    return dict2

dict2 = user_simlarity(user_data)


def Cl_recommender(Look_up_dict,hotels_dict,dict2, person, num_of_people):
    #find out the similar users
    dict_people = {}
    tmp1 = Look_up_dict[person].keys()
    for i in range(1, len(Look_up_dict)+1):
        tmp2 = Look_up_dict[str(i)].keys()
        tmp3 = set(tmp1).intersection(set(tmp2))
        if len(tmp3) > 0:
            dict_people[i] = list(tmp3)
    # The top_similar_users are firstly based on the number of intersect hotels, then based on the ratings for the hotels
    top_similar_users = []
    for i in sorted(dict_people, key=lambda k: (len(dict_people[k]), dict2[(person, str(k))], sum([Look_up_dict[person][s] for s in dict_people[k]])), reverse=True):
        top_similar_users.append([i, dict_people[i], dict2[(person, str(i))]])
    top_similar_users = top_similar_users[:num_of_people]
    # Further, we calcuate the weight based on users feature.
    #get the total likes for each hotel of the similar_users
    count_dict = {}
    for user in top_similar_users:
        for hotel in Look_up_dict[str(user[0])].keys():
            try:
                count_dict[hotel] += 1*user[2]
            except:
                count_dict[hotel] = 1*user[2]
    count_list = []
    #exclude the looked hotels
    for w in sorted(count_dict, key= lambda k: (count_dict[k], hotels_dict[str(k)] ) , reverse=True):
        if w in tmp1:
            continue
        else:
            count_list.append([w, count_dict[w]])
    # return the most likely
    return count_list[0][0]

if __name__ == "__main__":
    result = []
    for i in range(1,4545):
        result.append([i,Cl_recommender(Look_up_dict,hotels_dict,dict2, str(i), 10)])
    result = pd.DataFrame(result, columns=['user', 'hotel']
    result.to_csv('result_cl.txt',index=False,sep= '\t')




