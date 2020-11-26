import pandas as pd
import time
import numpy as np

def read_data():
    data=pd.read_csv("Diginetica.csv",sep=";")
    data=data.drop(["user_id","timeframe"],axis=1)
    return data

data=read_data()
timeframe=[]
date=data["eventdate"]
for x in date:
    # datex = time.mktime(time.strptime(x, '%Y-%m-%d'))
    datex = time.mktime(time.strptime(x, '%Y-%m-%d'))
    timeframe.append(datex)
data["eventdate"]=timeframe
header=["session_id:token","item_id:token","rating:float","timestamp:float"]

raintgs=np.ones_like(data["item_id"],dtype=int)
cloumns=data.columns.tolist()
cloumns.insert(2,"rating")
data=data.reindex(columns=cloumns)
data["rating"]=raintgs
# data=session_item_unique(data)

data.to_csv("diginetica.inter",sep="\t",index=False,header=header)