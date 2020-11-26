import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
def process_Diginetica():
    data=pd.read_csv("../origninal_data/Diginetica/Diginetica.CSV", sep=';',header=None, names=None, engine='python')
    data=data[1:]
    data=data.drop([1,4],axis=1)
    data=data.astype(int)
    data=data.values
    pickle.dump((data), open('../data/diginetica.txt', 'wb'))
# process_Diginetica()

def process_detail():
    data = pickle.load(open("../data/diginetica.txt","rb"))
    shuzhu_session=np.zeros(np.max(data[:,0])+1)
    shuzhu_item=np.zeros(np.max(data[:,1])+1)
    a=[]
    for data_i in data:
        shuzhu_item[data_i[1]]+=1
    for i in range(len(data)):
        if shuzhu_item[data[i][1]]<5:
            a.append(i)
    data=np.delete(data,a,axis=0)
    a=[]
    for data_i in data:
        shuzhu_session[data_i[0]]+=1
    for i in range(len(data)):
        if shuzhu_session[data[i][0]]<2:
            a.append(i)
    data=np.delete(data,a,axis=0)
    pickle.dump((data), open('../data/diginetica.txt', 'wb'))
# process_detail()

def process_details():
    data = pickle.load(open("../data/diginetica.txt", "rb"))
    data = sorted(data,key=lambda x: (x[0], x[-1]))
    data = np.array(data).astype(int)
    shuzhu_session = np.zeros(np.max(data[:, 0]) + 2).astype(int)
    session_item=defaultdict(set)
    for data_i in data:
        shuzhu_session[data_i[0]] += 1
    max_len=np.max(shuzhu_session)
    train_data = np.zeros((len(data) - len(set(data[:, 0])),max_len+1))
    for data_i in data:
        session_item[data_i[0]].add(data_i[1])
    lens=0
    for index in session_item:
        data_i=list(session_item[index])
        if len(data_i)>1:
            for i in range(len(data_i)-1):
                for j in range(i+1):
                    train_data[lens,j]=data_i[j]
                train_data[lens,-2]=data_i[i]
                train_data[lens,-1]=data_i[i+1]
                lens+=1
    train_data=np.array(train_data).astype(int)
    a=[]
    for i in range(len(train_data)):
        if all(train_data[i]==0):
            a.append(i)
    train_data = np.delete(train_data, a, axis=0)
    print(train_data[:1])
    pickle.dump((train_data), open('../data/diginetica_train.txt', 'wb'))
process_details()

