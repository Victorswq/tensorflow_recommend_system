import numpy as np
import tensorflow as tf

from data.dataset import Dataset
from sequential_model.AbstractRecommender import AbstractRecommender

class Caser(AbstractRecommender):
    def __init__(self,
                 learning_rate=0.001,
                 batch_size=512,
                 embedding_size=64,
                 episodes=100,
                 verbose=5,
                 data_name="ml_100k",
                 L=4,
                 window_size=[1,2,3,4],
                 filter_number=2,
                 negatives_num=2,
                 target_num=2):
        super(Caser,self).__init__()
        self.learning_rate=learning_rate

        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.episodes=episodes
        self.verbose=verbose
        self.data_name=data_name
        self.L=L
        self.widow_size=window_size
        self.filter_number=filter_number
        self.negative_num=negatives_num
        self.target_num=target_num

        self.dataset=Dataset(data_name=self.data_name)
        self.max_item_id=self.dataset.get_max_movie_id()+1
        self.max_user_id=self.dataset.get_max_user_id()+1

    def build_net(self):
        # placeholder
        self.user=tf.placeholder(tf.int32,shape=[None,1],name="user")
        self.item=tf.placeholder(tf.int32,shape=[None,self.L],name="item")
        self.pos_label=tf.placeholder(tf.int32,shape=[None,self.target_num],name="pos_label")
        self.neg_label=tf.placeholder(tf.int32,shape=[None,self.negative_num],name="neg_label")

        # matrix for embedding
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.max_item_id,self.embedding_size],initializer=init_w)
        self.user_matrix=tf.get_variable(name="user_matrix",shape=[self.max_user_id,self.embedding_size],initializer=init_w)

        # user_embedding and item_embedding
        self.item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item) # N * L * embedding_size
        self.user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user) # N * 1 * embedding_size

        # CNN and Max_pool
        pool_layer_last=[]
        with tf.name_scope("horizontal_CNN"):
            for i in range(len(self.widow_size)):
                filter_weights=tf.get_variable(name="filter_weight_%d"%self.widow_size[i],shape=[self.widow_size[i],self.embedding_size,1,self.filter_number],initializer=init_w)
                filter_bias=tf.get_variable(name="filter_bias_%d"%self.widow_size[i],shape=[self.filter_number],initializer=init_b)
                self.horizontal_CNN=tf.nn.conv2d(self.item_embedding[tf.newaxis,-1],filter=filter_weights,padding="valid",strides=[1,1,1,1],name="CNN_%d"%self.widow_size[i])
                # N * L * embedding_size * 1 ---> N * (L-widow_size+1) * 1 * filter_number
                self.horizontal_CNN=tf.nn.relu(tf.nn.bias_add(self.horizontal_CNN,filter_bias),name="rele_%d"%self.widow_size[i])
                self.horizontal_CNN=tf.nn.max_pool(self.horizontal_CNN,[1,self.L-self.widow_size[i]+1,1,1],strides=[1,1,1,1],padding="same",name="max_pool_%d"%self.widow_size[i])
                # N * (L-widow_size+1) * 1 * filter_number ---> N * 1 * 1 * filter_number
                pool_layer_last.append(self.horizontal_CNN)
            horizontal_CNN=tf.stack(pool_layer_last,axis=3) # N * 1 * 1 * len(window_size)_mul_filter_number

        pool_layer_last = []
        with tf.name_scope("vertical_CNN"):
            filter_weights=tf.get_variable(name="filter_weight_%d",shape=[1,self.L,1,self.filter_number],initializer=init_w)
            filter_bias=tf.get_variable(name="filter_bias_%d",shape=[self.filter_number],initializer=init_b)
            self.vertical_CNN=tf.nn.conv2d(tf.transpose(self.item_embedding[tf.newaxis,-1],[0,2,1,3]),filter=filter_weights,strides=[1,1,1,1],padding="valid",name="CNN")
            # N * embedding_size * L * 1 ---> N * (embedding_size) * 1 * filter_number
            self.vertical_CNN=tf.nn.relu(tf.nn.bias_add(self.vertical_CNN,filter_bias),name="relu_")
            self.vertical_CNN=tf.nn.max_pool(self.vertical_CNN,[1,self.embedding_size,1,1],strides=[1,1,1,1],padding="valid",name="max_pool")
            # N * (embedding_size) * 1 * filter_number ---> N * 1 * 1 * filter_number
            pool_layer_last.append(self.vertical_CNN)
            vertical_CNN=tf.stack(pool_layer_last,axis=1) # N * 1 * 1 * filter_number
        CNN=tf.concat([horizontal_CNN,vertical_CNN],axis=3) # N * 1 * 1 * (filter_number + len(window_size)_mul_filter_number)
        CNN=tf.squeeze(CNN) # N * (filter_number + len(window_size)_mul_filter_number)

        # fully connect
        self.z=tf.layers.dense(CNN,units=self.embedding_size,activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name="fully_connect_1")
        # N * (filter_number + len(window_size)_mul_filter_number) ---> N * embedding_size
        self.z=tf.concat([self.z,tf.squeeze(self.user_embedding)],axis=1) # N * (2_mul_embedding_size)
        self.z=tf.layers.dense(self.z,self.max_item_id,activation=None,name="prediction") # N * item_number

        # loss and train_op
        pos=self.z[self.pos_label]
        neg=self.z[self.neg_label]
        pos_loss=tf.reduce_mean(-tf.log(tf.nn.sigmoid(pos)+1e-20))
        neg_loss=tf.reduce_mean(-tf.log(tf.nn.sigmoid(1-neg)+1e-20))
        self.loss=pos_loss+neg_loss
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)