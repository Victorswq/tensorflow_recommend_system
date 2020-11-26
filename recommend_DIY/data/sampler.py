from data.dataset import Dataset
from data.data_process import Data
import numpy as np
from copy import deepcopy
import random


class PointSampler(object):
    def __init__(self,batch_size=512,data_name="ml_100k",num_neg=1):
        self.batch_size=batch_size
        self.data=Data()
        self.data_name=data_name
        self.num_neg=num_neg
        self.dataset = Dataset(data_name=self.data_name)

    def get_train_data(self,keep_label=False,value_for_negative=0.):
        self.data_value = self.data.get_train_data(data_name=self.data_name)
        user_movie_train=self.dataset.get_user_movie_for_train()
        num_item=self.dataset.get_max_movie_id()
        new_data_value=[]
        for user_item in self.data_value:
            user,item,label=user_item[0],user_item[1],user_item[2]
            if keep_label:
                new_data_value.append([user,item,label])
            else:
                new_data_value.append([user,item,1])
            for i in range(self.num_neg):
                j = np.random.choice(num_item)
                while j in user_movie_train[user]:
                    j = np.random.choice(num_item)
                new_data_value.append([user,j,value_for_negative])
        new_data_value=np.array(new_data_value)
        new_data_value[:,2].astype(np.float32)
        return new_data_value

    def get_train_batch(self,shuffle=False,keep_label=False,value_for_negative=0.):
        self.data_value_batch = self.get_train_data(keep_label=keep_label,value_for_negative=value_for_negative)
        if shuffle:
            index = [i for i in range(len(self.data_value_batch))]
            random.shuffle(index)
            self.data_value_batch = self.data_value_batch[index]
        for start in range(0,len(self.data_value_batch),self.batch_size):
            end=min(start+self.batch_size,len(self.data_value_batch))
            yield self.data_value_batch[start:end]

    def get_test_batch(self):
        test_data = self.data.get_test_data()
        len_test_data = len(test_data)
        batch_index = np.random.choice(len_test_data, size=self.batch_size)
        batch_data = test_data[batch_index, :]
        return batch_data

    def get_batch_number(self):
        i=1
        data_value = self.data.get_train_data(data_name=self.data_name)
        return (len(data_value))//self.batch_size*(i+self.num_neg)

class SamplerSeq(object):
    def __init__(self,batch_size=512,data_name="diginetica",shuffle=False):
        self.batch_size=batch_size
        self.data_name=data_name
        self.shuffle=shuffle
        self.data = Data()
        self.train_data = self.data.get_train_data(data_name=self.data_name)

    def get_train_batch(self):
        if self.shuffle:
            index=[i for i in range(len(self.train_data))]
            random.shuffle(index)
            self.train_data=self.train_data[index]
        for start in range(0,len(self.train_data),self.batch_size):
            end=min(start+self.batch_size,len(self.train_data))
            yield self.train_data[start:end]

    def get_batch_number(self):
        train_batch_number=len(self.train_data)//self.batch_size
        return train_batch_number+1

class SamplerFM(object):
    def __init__(self,batch_size=512,data_name="frappe",shuffle=False):
        self.batch_size=batch_size
        self.data_name=data_name
        self.shuffle=shuffle
        self.data=Data()
        self.train_data=self.data.get_train_data(data_name=self.data_name)
        self.validation_data=self.data.get_validation_data(data_name=self.data_name)
        self.test_data=self.data.get_test_data(data_name=self.data_name)

    def get_feature_size(self):
        max=0
        for i in range(len(self.train_data)):
            max_i=np.max(self.train_data[i][1:])
            if max_i>max:max=max_i
        return max+1

    def get_train_batch(self):
        if self.shuffle:
            index=[i for i in range(len(self.train_data))]
            random.shuffle(index)
            self.train_data=self.train_data[index]
        for start in range(0,len(self.train_data),self.batch_size):
            end=min(start+self.batch_size,len(self.train_data))
            yield self.train_data[start:end]

    def get_validation_batch(self):
        if self.shuffle:
            index=[i for i in range(len(self.validation_data))]
            random.shuffle(index)
            self.validation_data=self.validation_data[index]
        for start in range(0,len(self.validation_data),self.batch_size):
            end=min(start+self.batch_size,len(self.validation_data))
            yield self.validation_data[start:end]

    def get_test_len(self):
        return len(self.test_data)

    def get_validation_len(self):
        return len(self.validation_data)

    def get_train_len(self):
        return len(self.train_data)

    def get_test_batch(self):
        if self.shuffle:
            index=[i for i in range(len(self.test_data))]
            random.shuffle(index)
            self.test_data=self.test_data[index]
        for start in range(0,len(self.test_data),self.batch_size):
            end=min(start+self.batch_size,len(self.test_data))
            yield self.test_data[start:end]

    def get_batch_number(self):
        train_batch_number,validation_batch_number,test_batch_number=len(self.train_data)//self.batch_size,len(self.validation_data)//self.batch_size,len(self.test_data)//self.batch_size
        return train_batch_number+1,validation_batch_number+1,test_batch_number+1

class PairwiseSampler(object):
    def __init__(self,batch_size=512,data_name="ml_100k",num_neg=1):
        self.batch_size=batch_size
        self.data_name=data_name
        self.data=Data()
        self.num_neg=num_neg
        self.dataset=Dataset(data_name=self.data_name)

    def get_train_data(self):
        user_movie=self.dataset.get_user_movie()
        num_item=self.dataset.get_max_movie_id()
        data_value=self.data.get_train_data(data_name=self.data_name)
        for idx in range(len(data_value)):
            j=np.random.choice(num_item)+1
            while j in user_movie[data_value[idx,0]]:
                j = np.random.choice(num_item) + 1
            data_value[idx,2]=j
        return data_value

    def get_train_batch(self):
        data_value=self.get_train_data()
        for start in range(0,len(data_value),self.batch_size):
            end=min(start+self.batch_size,len(data_value))
            yield data_value[start:end]

    def get_test_batch(self):
        data_value = self.data.get_test_data(data_name=self.data_name)
        for start in range(0, len(data_value), self.batch_size):
            end = min(start + self.batch_size, len(data_value))
            yield data_value[start:end]

    def get_batch_number(self):
        data_value = self.data.get_train_data(data_name=self.data_name)
        return (len(data_value)+self.batch_size-1)//self.batch_size