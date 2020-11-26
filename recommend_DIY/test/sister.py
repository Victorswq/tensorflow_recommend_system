import pandas as pd


data=pd.read_csv("Immune.diffExp.txt",sep="\t")
values=data.values
print(values.shape)



"""

batch_size * seq_len * embedding_size
==>> GRU
batch_size * seq_len * hidden_size

1: repeat-explore mechanism
attentive: ==>>
batch_size * embedding_size
==>>: weight matrix
batch_size * 2

2: repeat recommendation decoder
batch_size * seq_len

3: explore recommendation decoder
batch_size * seq_len

"""