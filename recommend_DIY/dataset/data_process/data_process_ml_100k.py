import pickle
import pandas as pd

def get_test_data(number="1"):
    title = ['UserID', 'MovieID', 'Rating', 'time']
    test_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml-100k/u%s.test' % (number), sep='\t',
                              header=None, names=title, engine='python')
    train_data = pd.read_table('D:/PycharmProjects/recommend_DIY/dataset/origninal_data/ml-100k/u%s.base' % (number), sep='\t',
                              header=None, names=title, engine='python')

    test_data=test_data.values
    train_data=train_data.values

    pickle.dump((test_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_100k_test.txt', 'wb'))
    pickle.dump((train_data), open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_100k_train.txt', 'wb'))

get_test_data()


def test():
    data=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/ml_100k_test.txt', 'rb'))
    print(data)
# test()