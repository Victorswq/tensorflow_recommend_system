import pandas as pd

data=pd.read_csv("Diginetica.csv",sep=";")

fw = open("diginetica.csv", 'w')
pd.to_pickle(data,fw)