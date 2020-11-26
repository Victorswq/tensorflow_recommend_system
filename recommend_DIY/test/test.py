import pickle
import pandas as pd
import numpy as np
header=["session_id:token","item_id:token","rating:float","timestamp:float"]
data=pd.read_csv("Diginetica.csv",sep=";")
one=np.ones(data.shape[0],dtype=np.int)
col_name=data.columns.tolist()
col_name.insert(3,"rating")
data=data.reindex(columns=col_name)
data["rating"]=one
data=data.drop(["user_id","eventdate"],axis=1)
data=data.to_csv("diginetica.inter",sep="\t",index=False,header=header)