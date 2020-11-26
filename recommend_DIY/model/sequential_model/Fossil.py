import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model


class Fossil(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="Fossil"):
        super(Fossil, self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_named=data_name
        self.string=string

    def build_placeholder(self):
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.user_recent=tf.placeholder(tf.int32,[None,None],name="user_recent")
        self.num_idx=tf.placeholder(tf.float32,[None,],name="num_idx")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.item_recent=tf.placeholder(tf.placeholder,[None,None],name="item_recent")
        if self.is_pairwise is True:
            self.neg_user_recent=tf.placeholder(tf.int32,[None,None],name="neg_user_recent")
            self.neg_num_idx=tf.placeholder(tf.float32,[None,],name="neg_num_idx")
            self.num_neg_item=tf.placeholder(tf.int32,[None,],name="num_neg_item")
        else:
            self.label=tf.placeholder(tf.float32,[None,],name="label")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.user_matrix=tf.get