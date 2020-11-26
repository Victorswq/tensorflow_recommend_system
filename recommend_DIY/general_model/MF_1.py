import numpy as np
import tensorflow as tf
import time

from general_model.AbstractRecommender import AbstractRecommender
from data.dataset import Dataset
from data.sampler import PointSampler,PairwiseSampler
from evaluate.logger import Logger
from evaluate.evaluate import Evaluate


class MF(AbstractRecommender):
    def __init__(self,
                 sess,
                 learning_rate=0.001,
                 batch_size=512,
                 embedding_size=32,
                 is_pairwise=False,
                 data_name="ml_100k",
                 reg_mf=None,
                 episodes=100,
                 num_neg=1,
                 model_name="MF_1",
                 verbose=5,
                 ):
        super(MF,self).__init__(data_name=data_name,string=model_name)
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.is_pairwise=is_pairwise
        self.data_name=data_name
        self.reg_mf=reg_mf
        self.episodes=episodes
        self.num_neg=num_neg
        self.verbose=verbose

        self.sess=sess
        if self.is_pairwise is not True:
            self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=1)
        else:
            self.sampler=PairwiseSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=self.num_neg)
        self.build_graph()

    def build_graph(self):
        self.build_net()
        self.build_tools()

    def build_net(self):
        # placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        if self.is_pairwise is True:
            self.item_neg=tf.placeholder(tf.int32,[None,],name="item_neg")
        else:
            self.label=tf.placeholder(tf.float32,[None,],name="label")

        # matrix and embedding
        init_w=tf.random_uniform_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.item_matrix=tf.get_variable(name="user_matrix",shape=[self.num_items,self.embedding_size],dtype=tf.float32,initializer=init_w)
        self.user_matrix=tf.get_variable(name="item_matrix",shape=[self.num_users,self.embedding_size],dtype=tf.float32,initializer=init_w)
        self.item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item)
        self.user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user)
        if self.is_pairwise is True:
            self.neg_item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item_neg)

        # predict
        self.inference=tf.reduce_sum(tf.multiply(self.item_embedding,self.user_embedding),axis=1)
        if self.is_pairwise is True:
            neg_item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item_neg)
            neg_predict=tf.reduce_sum(tf.multiply(self.user_embedding,neg_item_embedding))
            self.loss=-tf.reduce_sum(tf.log_sigmoid(self.inference-neg_predict))
        else:
            self.loss=tf.reduce_mean(tf.square(self.inference-self.label))
        if self.reg_mf is not None:
            if self.is_pairwise is True:
                l2_loss=self.reg_mf*self.l2_loss(self.item_embedding,self.user_embedding,self.neg_item_embedding)
            else:
                l2_loss=self.reg_mf*self.l2_loss(self.item,self.user_embedding)
            self.loss+=l2_loss

        # optimizer
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        for episode in range(1,1+self.episodes):
            train_start_time=self.get_time()
            total_loss=0.0
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            if self.is_pairwise is not True:
                for i in range(batch_number):
                    data=next(data_iter)
                    loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.label:data[:,2]
                    })
                    total_loss+=loss
            else:
                for i in range(batch_number):
                    data=next(data_iter)
                    loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.item_neg:data[:,2],
                    })
                    total_loss+=loss
            self.logger.info("Episode %d:    loss: %.3f       running_time: %.3f"%(episode,total_loss/batch_number,self.get_time()-train_start_time))
            if episode%self.verbose==0:
                self.evaluate.F1(batch_size=400)

    def predict(self,user_ids,item_ids=None):
        ratings=[]
        if item_ids is None:
            for user_id in user_ids:
                items=np.arange(self.num_items)
                user=np.full(self.num_items,user_id)
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:user,
                    self.item:items
                })
                ratings.append(rating)
        else:
            pass
        return ratings

sess=tf.Session()
mf=MF(sess=sess,is_pairwise=True,reg_mf=0.1,learning_rate=0.0001)
sess.run(tf.global_variables_initializer())
mf.train()