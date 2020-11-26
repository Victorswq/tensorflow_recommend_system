import numpy as np
import pandas as pd


def get_session_item(data):
    values=data.values.astype(int)
    session_item={}
    for value in values:
        session=value[0]
        item=value[1]
        if session in session_item.keys():
            session_item[session]+=[item]
        else:
            session_item[session]=[item]
    return session_item

def get_item_session(data):
    values=data.values.astype(int)
    item_session={}
    for value in values:
        session=value[0]
        item=value[1]
        if item in item_session.keys():
            item_session[item]+=[session]
        else:
            item_session[item]=[session]
    return item_session

def remove_session_less_than_n(data,min_session=2):
    session_item=get_session_item(data=data)
    delete_list=[]
    values=data.values.astype(int)
    for idx,value in enumerate(values):
        session=value[0]
        if len(session_item[session])<min_session:
            delete_list.append(idx)
    data=data.drop(delete_list,axis=0)
    data=data.reset_index(drop=True)
    return data

def remove_item_less_than_n(data,min_item=5):
    item_session=get_item_session(data=data)
    delete_list=[]
    values=data.values.astype(int)
    for idx,value in enumerate(values):
        item=value[1]
        if len(item_session[item])<min_item:
            delete_list.append(idx)
    data=data.drop(delete_list,axis=0)
    data=data.reset_index(drop=True)
    return data

def session_item_unique(data):
    columns=("session_id:token","item_id:token","rating:float","timestamp:float")
    data = data.reindex(columns=columns)
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
    data["session_id:token"]=sessions
    data["item_id:token"]=items
    return data

def check(data):
    session_item=get_session_item(data)
    item_session=get_item_session(data)
    for key,value in session_item.items():
        if len(value)<2:
            print("You losser!!!!")
            return
    for key,value in item_session.items():
        if len(value)<5:
            print("You losser!!!")
            return

def data_graph(data,min_session_n=2,mint_item_n=5):
    data=session_item_unique(data)
    data=remove_session_less_than_n(data,min_session_n)
    data = session_item_unique(data)
    data=remove_item_less_than_n(data,mint_item_n)
    return data

def putting_rating(data):
    ratings = np.ones_like(data["item_id"], dtype=np.int)
    col_name = data.columns.tolist()
    col_name.insert(2, "rating")
    data = data.reindex(columns=col_name)
    data["rating"] = ratings
    return data

def get_data_diginetica(min_session_n=2,min_item_n=5,min_iter_num=5):
    data=pd.read_csv("Diginetica.csv",sep=";")
    data=data.drop(["user_id","eventdate"],axis=1)
    for i in range(min_iter_num):
        data=data_graph(data=data,min_session_n=min_session_n,mint_item_n=min_item_n)
        print(data)
        check(data)
    data=putting_rating(data)
    values = data.values
    print(np.max(values[:, 1]))
    print(np.unique(values[:, 1]))
    header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
    data.to_csv("diginetica.inter", sep="\t", index=False, header=header)

# get_data(min_iter_num=8)
def get_data_yoochoose(min_session_n=2,min_item_n=5,min_iter_num=5):
    data=pd.read_table("yoochoose-clicks.dataset",sep="\t")
    for i in range(min_iter_num):
        data=data_graph(data=data,min_session_n=min_session_n,mint_item_n=min_item_n)
        print(data)
        check(data)
    values = data.values
    print(np.max(values[:, 1]))
    print(np.unique(values[:, 1]))
    header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
    data.to_csv("yoochoose.inter", sep="\t", index=False, header=header)

get_data_yoochoose(min_iter_num=10)