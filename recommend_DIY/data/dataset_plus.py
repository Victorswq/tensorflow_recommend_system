import os
import pandas as pd
from scipy.sparse import csr_matrix
from util.tool import csr_to_user_dict_bytime,csr_to_user_dict

class Dataset_plus(object):
    def __init__(self,
                 data_name="ml_100k",
                 ):
        self.train_matrix=None
        self.test_matrix=None
        self.time_matrix=None
        self.negative_matrix=None
        self.userids=None
        self.itemids=None
        self.num_users=None
        self.num_items=None
        self.dataset_name=data_name
