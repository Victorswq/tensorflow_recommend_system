from model.AbstractRecommender import Sequential_Model

ss=Sequential_Model(data_name="diginetica")
a,s,c=ss.generate_sequences()

for aa,cc in zip(a,c):
    print(aa," >>>>>>> ",cc)

# import numpy as np
# a=-np.inf
# print(a+np.inf)