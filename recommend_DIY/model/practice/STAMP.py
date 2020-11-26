import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model
from util.data_iterator import DataIterator


class STAMP(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="diginetica",
                 string="STAMP",
                 ):
        super(STAMP,self).__init__(data_name,string)
        self.sess=sess
        self.data_name=data_name
        self.string=string

    def build_placeholder(self):
        self.seq_item=tf.placeholder(tf.int32,[None,self.max_len],name="seq_item_input")
        self.seq_label=tf.placeholder(tf.int32,[None,],name="seq_label")
        self.seq_mask=tf.placeholder(tf.float32,[None,self.max_len],name="seq_mask")

    def build_variables(self):
        init_w1=tf.random_normal_initializer(0.,0.01)
        init_w2=tf.random_normal_initializer(0.,0.05)
        init_b=tf.constant_initializer(0.)
        pad_for_zero=tf.zeros([1,self.embedding_size],dtype=tf.float32)
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w1)
        self.item_matrix=tf.concat([pad_for_zero,self.item_matrix[1:]],axis=0)
        self.w1=tf.get_variable(name="w1_for_ms",shape=[self.embedding_size,self.embedding_size],initializer=init_w1)
        self.w2=tf.get_variable(name="w2_for_last_click",shape=[self.embedding_size,self.embedding_size],initializer=init_w1)
        self.w3=tf.get_variable(name="w3_for_click",shape=[self.embedding_size,self.embedding_size],initializer=init_w1)
        self.layer_ms=tf.layers.Dense(self.embedding_size,activation=tf.nn.tanh,kernel_initializer=init_w2,bias_initializer=init_b)
        self.layer_mt=tf.layers.Dense(self.embedding_size,activation=tf.nn.tanh,kernel_initializer=init_w2,bias_initializer=init_b)

    def build_inference(self,item_input):
        pass

    def build_graph(self):
        batch_size=tf.shape(self.seq_item)[0]
        self.seq_item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.seq_item,name="item_embedding_for_item_seq")
        # N * max_len * embedding_size
        mean_seq_item=tf.reduce_mean(self.seq_item_embedding,axis=1)
        # N * embedding_size
        last_click=self.seq_item_embedding[:,-1,:]
        # N * embedding_size
        w13d=tf.reshape(tf.tile(self.w1,[batch_size,1]),(batch_size,self.embedding_size,self.embedding_size))
        w1ms=tf.matmul(mean_seq_item,w13d)
        # N *