import pandas as pd
import numpy as np
import time

def read_numd_data(num=515687):
    header = ["session_id", "date","item_id", "type"]
    data = pd.read_csv("yoochoose-clicks.dat", sep=",", nrows=num, header=None)
    data.to_csv("yoochoose-clicks.inte", sep="\t", index=False, header=header)

def preocess_data():
    data=pd.read_csv("yoochoose-clicks.inte",sep="\t")
    data = data.drop("type", axis=1)
    date_data=data["date"]
    dates=np.zeros(data.shape[0])
    for idx,date in enumerate(date_data):
        dates[idx]=time.mktime(time.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ'))
    header=["session_id:token","item_id:token","rating:float","timestamp:float"]
    one=np.ones(data.shape[0],dtype=np.int)
    col_name=data.columns.tolist()
    col_name.insert(2,"rating")
    data=data.reindex(columns=col_name)
    data["date"]=dates
    data["rating"]=one
    data_1=data["date"]
    data["date"]=data["item_id"]
    data["item_id"]=data_1
    data_session_id=np.array(data["session_id"])
    data_item_id=np.array(data["date"])
    data_session_id_dict={}
    data_item_id_dict={}
    ctr=1
    ctrr=1
    for i in range(len(data_session_id)):
        if data_session_id[i] in data_session_id_dict.keys():
            data_session_id[i]=data_session_id_dict[data_session_id[i]]
        else:
            data_session_id_dict[data_session_id[i]]=ctr
            data_session_id[i]=ctr
            ctr+=1
    for i in range(len(data_item_id)):
        if data_item_id[i] in data_item_id_dict.keys():
            data_item_id[i]=data_item_id_dict[data_item_id[i]]
        else:
            data_item_id_dict[data_item_id[i]]=ctrr
            data_item_id[i]=ctrr
            ctrr+=1
    data["session_id"]=data_session_id
    data["date"]=data_item_id
    header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
    data.to_csv("yoochoose-clicks.inter",sep="\t",index=False,header=header)

# read_numd_data()
preocess_data()