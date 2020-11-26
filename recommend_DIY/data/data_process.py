import pandas as pd
import numpy as np
import pickle
class Data(object):
    def __init__(self):
        pass

    def get_test_data(self,data_name="diginetica"):
        data_name=data_name+"_test.txt"
        test_data = pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/%s'%(data_name),"rb"))
        return test_data

    def get_train_data(self,data_name="diginetica"):
        data_name = data_name + "_train.txt"
        train_data = pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/%s' % (data_name), "rb"))
        return train_data

    def get_validation_data(self,data_name="diginetica"):
        data_name = data_name + "_validation.txt"
        train_data = pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/%s' % (data_name), "rb"))
        return train_data

    def get_max_user(self,data_name="diginetica"):
        data_name=data_name+"_max_user.txt"
        max_user=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/%s' % (data_name), "rb"))
        return max_user

    def get_max_item(self,data_name="diginetica"):
        data_name=data_name+"_max_item.txt"
        max_item=pickle.load(open('D:/PycharmProjects/recommend_DIY/dataset/data/%s' % (data_name), "rb"))
        return max_item

# data=Data()
# # datas=data.get_test_data()
# datas=data.get_train_data()[0]
# max_item=0
# print(datas[1])
# for da in datas:
#     print(da)
#     if max_item < len(da):
#         max_item=len(da)
# print(max_item)