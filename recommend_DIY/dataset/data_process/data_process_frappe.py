import pickle
import pandas as pd

def get_data_as_we_want(data):
    for i in range(len(data)):
        data_i=data[i]
        for j in range(1,len(data_i)):
            data_i[j]=int(data_i[j][:-2])
    return data

def get_test_data():
    test_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/frappe/frappe.test.libfm', sep=' ',
                              header=None, names=None, engine='python')
    train_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/frappe/frappe.train.libfm', sep=' ',
                              header=None, names=None, engine='python')
    validation_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/frappe/frappe.validation.libfm',sep=' ',
                              header=None, names=None, engine='python')
    test_data=test_data.values
    train_data=train_data.values
    validation_data=validation_data.values

    train_data=get_data_as_we_want(train_data)
    validation_data=get_data_as_we_want(validation_data)
    test_data=get_data_as_we_want(test_data)
    pickle.dump((test_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/frappe_test.txt', 'wb'))
    pickle.dump((train_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/frappe_train.txt', 'wb'))
    pickle.dump((validation_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/frappe_validation.txt', 'wb'))

get_test_data()

def test():
    data=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/frappe_test_data', 'rb'))
    print(data)
# test()