from data.data_process import Data
import numpy as np
from collections import defaultdict


class Dataset(object):
    """
    number:the number of the train/test data
    """
    def __init__(self,data_name="ml_100k"):
        self.data_name=data_name
        self.data=Data()

    def get_max_user_id(self):
        train_data=self.data.get_train_data(data_name=self.data_name)
        return np.max(train_data[:,0])

    def get_max_movie_id(self):
        train_data=self.data.get_train_data(data_name=self.data_name)
        return np.max(train_data[:,1])

    def get_max_len(self):
        train_data=self.data.get_train_data(data_name=self.data_name)
        return len(train_data[0])

    def get_feature_size(self):
        pass

    def get_feature_max_len(self):
        pass

    def get_user_movie(self):
        user_movie=defaultdict(set)
        train_data = self.data.get_test_data(data_name=self.data_name)
        length=len(train_data)
        for idx in range(length):
            user_movie[train_data[idx,0]].add(train_data[idx,1])
        return user_movie

    def get_user_movie_for_train(self):
        user_movie=defaultdict(set)
        train_data = self.data.get_train_data(data_name=self.data_name)
        length=len(train_data)
        for idx in range(length):
            user_movie[train_data[idx,0]].add(train_data[idx,1])
        return user_movie

    def get_user_movie_for_test(self):
        user_movie=defaultdict(set)
        test_data = self.data.get_test_data(data_name=self.data_name)
        length=len(test_data)
        for idx in range(length):
            user_movie[test_data[idx,0]].add(test_data[idx,1])
        return user_movie

    def get_max_movie_id_for_test(self):
        train_data=self.data.get_test_data(data_name=self.data_name)
        return np.max(train_data[:,1])

    def get_max_user_id_for_test(self):
        train_data=self.data.get_test_data(data_name=self.data_name)
        return np.max(train_data[:,0])

# dataset=Dataset(data_name="diginetica")
# print(dataset.get_max_len())