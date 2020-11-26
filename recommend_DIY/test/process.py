import pandas as pd
import numpy as np


data=pd.read_csv("train-clicks.csv",sep=";")
print(len(np.unique(data["queryId"])))