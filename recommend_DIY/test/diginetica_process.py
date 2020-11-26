import pandas as pd
import numpy as np

def get_item_session_unique(data):
    values=data.values.astype(int)
    session_dict={}
    item_dict={}
    session_start=1
    item_start=1
    data = data.reset_index(drop=True)
    new_sessions=np.zeros_like(values[:,0],dtype=np.int)
    new_items=np.zeros_like(values[:,1],dtype=np.int)
    for idx,value in enumerate(values):
        session=value[0]
        item=value[1]
        # change for session
        if session in session_dict.keys():
            new_sessions[idx]=session_dict[session]
        else:
            new_sessions[idx]=session_start
            session_dict[session]=session_start
            session_start+=1

        # change for item
        if item in item_dict.keys():
            new_items[idx]=item_dict[item]
        else:
            new_items[idx]=item_start
            item_dict[item]=item_start
            item_start+=1

    data["session_id"]=new_sessions
    data["item_id"]=new_items
    return data

def get_session_dict(data):
    session_item_dict={}
    session_index_dict={}
    values=data.values.astype(int)
    for idx,value in enumerate(values):
        session=value[0]
        item=value[1]
        item_list=[]
        index_list=[]
        if session in session_item_dict.keys():
            item_list=session_item_dict[session]
            index_list=session_index_dict[session]
        session_item_dict[session]=[item]+item_list
        session_index_dict[session]=[idx]+index_list
    return session_index_dict,session_item_dict

def get_item_dict(data):
    item_session_dict={}
    values=data.values.astype(int)
    for value in values:
        session=value[0]
        item=value[1]
        session_list=[]
        if item in item_session_dict.keys():
            session_list=item_session_dict[item]
            item_session_dict[item] = [session] + session_list
        else:
            item_session_dict[item] = [session]
    return item_session_dict

def get_session_less_than_n(data,n=2):
    session_index_dict,session_item_dict=get_session_dict(data=data)
    values=data.values.astype(int)
    delete_list=[]
    for idx,value in enumerate(values):
        session=value[0]
        session_item_list=list(session_item_dict[session])
        if len(session_item_list)<n:
            delete_list.append(idx)
    data=data.drop(delete_list,axis=0)
    return data

def get_item_less_than_n(data,n_item=5,n_session=2):
    item_session_dict=get_item_dict(data=data)
    session_index_dict, session_item_dict=get_session_dict(data=data)
    values=data.values.astype(int)
    delete_list=[]
    for idx,value in enumerate(values):
        item=value[1]
        session_list=list(item_session_dict[item])
        if len(session_list)<n_item:
            delete_list.append(idx)
            for session in session_list:
                item_list=list(session_item_dict[session])
                if len(item_list)>0:
                    item_list.pop()
                session_item_dict[session]=item_list
                item_list=session_item_dict[session]
                if len(item_list)<n_session:
                    index=list(session_index_dict[session])
                    delete_list+=index
    delete_list=np.unique(np.array(delete_list,dtype=np.int))
    data=data.drop(delete_list,axis=0)
    return data

def process_graph(data,min_n_item=5,min_n_session=2):
    data=get_item_session_unique(data=data)
    data=get_session_less_than_n(data=data,n=min_n_session)
    data=get_item_session_unique(data=data)
    data=get_item_less_than_n(data=data,n_item=min_n_item,n_session=min_n_session)
    data=get_item_session_unique(data)
    return data

def get_data(min_n_item=5,min_n_session=2):
    data=pd.read_csv("Diginetica.csv",sep=";")
    data=data.drop(["user_id","eventdate"],axis=1)
    data=process_graph(data=data,min_n_item=min_n_item,min_n_session=min_n_session)
    data=putting_rating(data=data)
    values=data.values
    print(np.max(values[:,1]))
    print(data)
    header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
    data.to_csv("diginetica.inter", sep="\t", index=False, header=header)

def putting_rating(data):
    ratings = np.ones_like(data["item_id"], dtype=np.int)
    col_name = data.columns.tolist()
    col_name.insert(2, "rating")
    data = data.reindex(columns=col_name)
    data["rating"] = ratings
    return data

# get_data()



def process():
    data=pd.read_csv("Diginetica.csv",sep=";")
    data=data.drop(["user_id","eventdate"],axis=1)

    values = data.values.astype(np.int)
    print(len(values[:, 0]))
    num_per_items=np.zeros(shape=(np.max(np.array(data["item_id"],dtype=np.int))+1,),dtype=np.int)
    numbers = np.zeros(shape=(np.max(np.array(data["item_id"], dtype=np.int)) + 1,), dtype=np.int)
    for item in data["item_id"]:
        num_per_items[item]+=1
    for idx,item in enumerate(num_per_items):
        if item<5 and item>0:
            numbers[idx]=1
    print(np.sum(numbers))
    data_delete=[]
    for idx,v in enumerate(numbers):
        if v == 1:
            data_delete.append(idx)
    # values=data.values
    # for idx,value in enumerate(values):
    #     if value[1] in item_delete:
    #         data_delete.append(idx)
    print(data_delete)
    print(len(data_delete))
    data=data.drop(data_delete,axis=0)

    ratings=np.ones_like(data["item_id"],dtype=np.int)
    col_name = data.columns.tolist()
    col_name.insert(2, "rating")
    data=data.reindex(columns=col_name)
    data["rating"]=ratings

    user_id={}
    item_id={}
    values=data.values.astype(np.int)
    user_start=1
    item_start=1
    user=np.zeros_like(values[:,0],dtype=np.int)
    item=np.zeros_like(values[:,0],dtype=np.int)
    for idx,value in enumerate(values):
        if value[0] in user_id.keys():
            user[idx]=user_id[value[0]]
        else:
            user[idx]=user_start
            user_id[value[0]]=user_start
            user_start+=1

        if value[1] in item_id.keys():
            item[idx]=item_id[value[1]]
        else:
            item[idx]=item_start
            item_id[value[1]]=item_start
            item_start+=1
    data["item_id"]=item
    data["session_id"]=user
    values=data.values.astype(np.int)
    print(len(values[:, 0]))
    header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
    data.to_csv("diginetica.inter",sep="\t",index=False,header=header)

# process()