import pickle
import pandas as pd
from collections import defaultdict
import numpy as np
from data.data_process import Data

data_name="ml_1m_lunwen"
data=Data()

def get_test_data():
    title = ['UserID', 'MovieID', 'Rating', 'time']
    test_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml_1m_lunwen/ml-1m.test.rating', sep='\t',
                              header=None, names=title, engine='python')
    train_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml_1m_lunwen/ml-1m.train.rating', sep='\t',
                              header=None, names=title, engine='python')
    test_data=test_data.values
    train_data=train_data.values
    item_ids = defaultdict(set)
    pickle.dump((test_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_1m_lunwen_test.txt', 'wb'))
    pickle.dump((train_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_1m_lunwen_train.txt', 'wb'))

get_test_data()

def get_NeuMF_test_data():
    title = ['UserID', 'MovieID', 'Rating', 'time']
    test_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml_1m_lunwen/ml-1m.test.rating',
                              sep='\t',
                              header=None, names=title, engine='python')
    data_test = pd.read_table(
        'D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml_1m_lunwen/ml-1m.test.negative', sep='\t',
        header=None, names=None, engine='python')
    data_test = data_test.values
    test_data = test_data.values
    data_test[:, 0] = data_test[:, -1]
    data_test[:, -1] = test_data[:, 1]
    pickle.dump(data_test, open("D:/PycharmProjects/recommend_DIY/dataset/evaluate/%s.d" % data_name, mode="wb"))
# get_NeuMF_test_data()


def test():
    data=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_100k_test.txt', 'rb'))
    print(data)
# test()