import pandas as pd
import numpy as np

def session_item_unique(data):
    values=data.values.astype(int)
    session_dict={}
    item_dict={}
    session_start=1
    item_start=1
    sessions=[]
    items=[]
    for idx,value in enumerate(values):
        session=value[0]
        item=value[1]
        if session in session_dict.keys():
            sessions.append(session_dict[session])
        else:
            session_dict[session]=session_start
            sessions.append(session_dict[session])
            session_start+=1

        if item in item_dict.keys():
            items.append(item_dict[item])
        else:
            item_dict[item]=item_start
            items.append(item_dict[item])
            item_start+=1
    data["session_id"]=sessions
    data["item_id"]=items
    return data

data=pd.read_csv("Diginetica.csv",sep=";")
data=data.drop(["eventdate","user_id"],axis=1)
header=["session_id:token","item_id:token","rating:float","timestamp:float"]

raintgs=np.ones_like(data["item_id"],dtype=int)
cloumns=data.columns.tolist()
cloumns.insert(2,"rating")
data=data.reindex(columns=cloumns)
data["rating"]=raintgs
# data=session_item_unique(data)

data.to_csv("diginetica.inter",sep="\t",index=False,header=header)

print(data)