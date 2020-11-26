import tensorflow as tf
import numpy as np
import pickle

from data import PointSampler
from data import Dataset
from evaluate.evaluate import Evaluate
from evaluate.logger import Logger
from time import time as timee
import time


class MF(object):
    def __init__(self,
                 sess,
                 embedding_size=64,
                 batch_size=1024,
                 episodes=1000,
                 learning_rate=0.0005,
                 reg_mf=8,
                 data_name="ml_100k",
                 verbose=1,
                 num_neg=1):
        self.sess=sess
        self.embedding_size=embedding_size
        self.lr=learning_rate
        self.reg_mf=reg_mf
        self.batch_size=batch_size
        self.episodes=episodes
        self.data_name=data_name
        self.verbose=verbose
        self.num_neg=num_neg

        # self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=self.num_neg)
        self.sampler = PointSampler(batch_size=self.batch_size, data_name=self.data_name, num_neg=self.num_neg)
        self.dataset=Dataset(data_name=self.data_name)
        self.string="MF"

        self.build_net()
        self.var = [v.name for v in tf.trainable_variables()]

    def build_net(self):
        #placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.label=tf.placeholder(tf.float32,[None,],name="label")
        #variable
        self.user_matrix=tf.get_variable("user_matrix",[self.dataset.get_max_user_id()+1,self.embedding_size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.1))
        self.item_matrix=tf.get_variable("item_matrix",[self.dataset.get_max_movie_id()+1,self.embedding_size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.1))

        self.user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user)
        self.item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item)

        #predict
        # self.inference=tf.reduce_sum(tf.multiply(self.user_embedding,self.item_embedding),axis=1)
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.inference = tf.multiply(self.user_embedding, self.item_embedding)
        inputs=tf.concat([self.user_embedding,self.item_embedding],axis=1)
        # self.inference = tf.layers.dense(self.inference,units=1,activation=None,kernel_initializer=tf.random_normal_initializer(1,0.1))
        # self.inference = tf.reduce_mean(self.inference,axis=1)
        self.weight = tf.layers.dense(inputs,units=self.embedding_size,activation=tf.nn.relu,name="factor_weight_1",kernel_regularizer=regularizer)
        self.weight = tf.layers.dense(self.weight, units=self.embedding_size, activation=tf.nn.softmax, name="factor_weight",kernel_regularizer=regularizer)
        self.inference = tf.reduce_sum(tf.multiply(self.inference,self.weight),axis=1)
        add_loss = tf.losses.get_regularization_loss()
        #loss
        self.loss=tf.losses.mean_squared_error(labels=self.label,predictions=self.inference)
        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding, self.user_embedding)),
            tf.reduce_sum(tf.multiply(self.item_embedding, self.item_embedding)),
        ])
        # self.loss+=self.reg_mf*l2_norm/1000+self.reg_mf*add_loss
        self.loss += self.reg_mf * add_loss
        #optimizer
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def train(self):
        """
                training and store the log of the training
                """
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/MF/%s" % str(localtime))
        self.evaluate = Evaluate(dataset=self.dataset, data_name=self.data_name, logger=self.logger, model=self)
        self.evaluate.logging()
        for episode in range(self.episodes):
            total_loss=0
            train_start_time=timee()
            # data_iter=self.sampler.get_train_batch()
            data_iter = self.sampler.get_train_batch()
            batch_num=self.sampler.get_batch_number()
            for batch_i in range(batch_num):
                data=next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.user:data[:,0],
                    self.item:data[:,1],
                    self.label:data[:,2]
                })
                total_loss+=loss
            self.logger.info("--Episode %d:    loss : %f    time : %f"%(episode,total_loss/batch_num,timee()-train_start_time))
            if (episode+1)%self.verbose==0:
                if self.data_name=="ml_100k":
                    self.evaluate.RMSE(batch_size=self.batch_size)
                    self.evaluate.F1(top_k=[20,40,60,80],batch_size=400)
                else:
                    self.evaluate.HR(top_k=[1,2,10,20],batch_size=400)
                    self.evaluate.NDCG(top_k=[10,20],batch_size=400)
        """
        store the model
        """
        values = self.sess.run(self.var)
        pickle.dump((values), open('D:/PycharmProjects/recommend_DIY/dataset/mf.p', 'wb'))

    def predict(self,user_ids,items=None):
        user_matrix,item_matrix=self.sess.run([self.user_matrix,self.item_matrix])
        user_matrix=user_matrix[user_ids]
        ratings=np.matmul(user_matrix,item_matrix.T)
        if items is not None:
            ratings=[rating[item] for rating,item in zip(ratings,items)]
        return ratings

sess=tf.Session()
mf=MF(sess,learning_rate=0.001,data_name="ml_100k",verbose=10,reg_mf=1)
sess.run(tf.global_variables_initializer())
mf.train()