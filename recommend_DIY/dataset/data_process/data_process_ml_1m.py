import pickle
import pandas as pd
from collections import defaultdict
import numpy as np
from data.data_process import Data

data_name="ml_1m"
data=Data()

def get_test_data():
    user_item=defaultdict(set)
    user_time=defaultdict(set)

    test_data=[]
    train_data=[]

    title = ['UserID', 'MovieID', 'Rating', 'time']
    data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml-1m/ratings.dat', sep='::',
                              header=None, names=title, engine='python')
    data=data.values
    for idx in range(len(data)):
        user_item[data[idx,0]].add(data[idx,1])
        user_time[data[idx,0]].add(data[idx,3])


    for user in user_time:
        idx=np.argmax(list(user_time[user]))
        item=list(user_item[user])[idx]
        test_data.append([user,item,1.,])
        for item_id in user_item[user]:
            if item_id !=item:
                train_data.append([user,item_id,1.])
    test_data=np.array(test_data,dtype=int)
    train_data=np.array(train_data,dtype=int)
    test_data[:,2].astype(float)
    train_data[:,2].astype(float)

    pickle.dump((test_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_1m_test.txt', 'wb'))
    pickle.dump((train_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_1m_train.txt', 'wb'))

# get_test_data()


def get_NeuMF_test_data():
    train_data = data.get_train_data(data_name=data_name)
    item_ids = defaultdict(set)
    test_data = data.get_test_data(data_name=data_name)
    test_user_item=defaultdict(set)
    for i in range(len(test_data)):
        test_user_item[test_data[i,0]].add(test_data[i,1])
    for user in test_user_item:
        for i in range(99):
            j = np.random.choice(test_data[:, 1], size=1)[0]
            while j in train_data[user] or j in item_ids[user] or j in test_user_item[user]:
                j = np.random.choice(test_data[:, 1], size=1)[0]
            item_ids[user].add(j)
        item_ids[user].add(list(test_user_item[user])[0])
    pickle.dump(item_ids,open("D:/PycharmProjects/recommend_DIY/dataset/evaluate/%s.d"%data_name,mode="wb"))
get_NeuMF_test_data()

def test():
    data=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_1m_test.txt', 'rb'))
    print(data[:100])
# test()