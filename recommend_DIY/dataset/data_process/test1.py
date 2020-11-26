import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from util.tool import csr_to_user_dict_bytime,csr_to_user_dict
import pickle

def get_test_data():
    """
    session_id;user_id;item_id;timeframe;eventdate
    1;         NA;     81766;  526309;   2016-05-09
    """
    train_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/Diginetica/Diginetica.csv', sep=';',
                              header=None, names=None, engine='python')
    train_data= train_data[1:]
    train_data = train_data.drop([1,4], axis=1)
    train_data=train_data.astype(int)
    train_data=train_data.values
    # print(len(train_data))
    train_matrix=csr_matrix((np.ones_like(train_data[:,2]),(train_data[:,0],train_data[:,1])),shape=(np.max(train_data[:,0]).astype(int)+1,np.max(train_data[:,1]).astype(int)+1))
    # print(train_matrix)
    time_matrix = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])),
                              shape=(np.max(train_data[:, 0]).astype(int) + 1, np.max(train_data[:, 1]).astype(int) + 1))
    # print(time_matrix)
    # print(time_matrix.todok())
    final=csr_to_user_dict_bytime(time_matrix,train_matrix)
    pops=[]
    max_user=0
    max_item=0
    for u,items in final.items():
        if len(items)<3:
            pops.append(u)
        else:
            max_user = u
            max_item=np.max(items)

    max_user+=1
    max_item+=1
    for u in pops:
        final.pop(u)
    train={}
    test={}
    for u,items in final.items():
        train[u]=items[:-1]
        test[u]=items[-1]

    pickle.dump((final), open('D:/PycharmProjects/recommend_DIY/dataset/first_step_process/diginetica/diginetica_train.txt', 'wb'))
    pickle.dump((test),open('D:/PycharmProjects/recommend_DIY/dataset/first_step_process/diginetica/diginetica_test.txt', 'wb'))
    pickle.dump((max_user),
                open('D:/PycharmProjects/recommend_DIY/dataset/first_step_process/diginetica/diginetica_max_user.txt',
                     'wb'))
    pickle.dump((max_item),
                open('D:/PycharmProjects/recommend_DIY/dataset/first_step_process/diginetica/diginetica_max_item.txt',
                     'wb'))
    # print(train_matrix)
    # print("ok")
    # print(train_matrix[2])
    # print("yes")
    # print(train_matrix[1])
    # print(train_matrix)

# get_test_data()


def test():
    data=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_100k_test.txt', 'rb'))
    print(data)
# test()