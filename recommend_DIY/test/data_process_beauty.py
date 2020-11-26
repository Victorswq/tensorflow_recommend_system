import numpy as np
import pandas as pd


def session_item_unique(data):
    columns = ("user_id:token", "item_id:token", "rating:float", "timestamp:float")
    data.columns = columns
    values = data.values
    session_dict = {}
    item_dict = {}
    session_start = 1
    item_start = 1
    sessions = []
    items = []
    count=0
    for idx, value in enumerate(values):
        count+=1
        session = value[0]
        item = value[1]
        if session in session_dict.keys():
            sessions.append(session_dict[session])
        else:
            session_dict[session] = session_start
            sessions.append(session_dict[session])
            session_start += 1

        if item in item_dict.keys():
            items.append(item_dict[item])
        else:
            item_dict[item] = item_start
            items.append(item_dict[item])
            item_start += 1
    print("all thing is ",count)
    data["user_id:token"] = sessions
    data["item_id:token"] = items
    return data


# data=pd.read_csv("ratings_Beauty.csv",sep=",")
# data=session_item_unique(data)
# data.to_csv("beauty.inter",sep="\t",index=False)

data=pd.read_csv("ml_1m.dat",sep="::",)
data=session_item_unique(data)
data.to_csv("ml-1m.inter",sep="\t",index=False)