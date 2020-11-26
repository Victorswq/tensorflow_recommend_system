import pandas as pd
import numpy as np


data=pd.read_csv("Diginetica.csv",sep=";")
data=data.drop(["user_id","eventdate"],axis=1)
ratings=np.ones_like(data["item_id"],dtype=np.int)
col_name = data.columns.tolist()
col_name.insert(2, "rating")
data=data.reindex(columns=col_name)
data["rating"]=ratings
header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
data.to_csv("diginetica",sep="\t",index=False,header=header)