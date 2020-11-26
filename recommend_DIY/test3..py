import numpy as np


a=-np.random.uniform()
for i in range(1000):
    a=1+1/a
    print(a)