import tensorflow as tf
import pickle
import time
from time import time as timee
import numpy as np

from data.sampler import PairwiseSampler
from data.dataset import Dataset
from evaluate.evaluate import Evaluate
from evaluate.logger import Logger


class BPR(object):
    def __init__(self,
                 sess,
                 learning_rate=0.0001,
                 reg_mf=0.0001,
                 batch_size=512,
                 embedding_size=160,
                 number="1",
                 episodes=1000,
                 verbose=10,
                 num_neg=1,
                 ):
        self.sess=sess
        self.lr=learning_rate
        self.reg_mf=reg_mf
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.number=number
        self.episodes=episodes
        self.verbose=verbose
        self.num_neg=num_neg

        self.dataset=Dataset(number=self.number)
        self.sampler=PairwiseSampler(batch_size=self.batch_size,number=self.number)

        self.build_net()
        self.var=[v.name for v in tf.trainable_variables()]

    def build_net(self):
        #placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item_pos=tf.placeholder(tf.int32,[None],name="item_pos")
        self.item_neg=tf.placeholder(tf.int32,[None],name="item_neg")

        #variables
        self.user_matrix=tf.get_variable("user_matrix",[self.dataset.get_max_user_id()+1,self.embedding_size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.1))
        self.item_matrix=tf.get_variable("item_matrix",[self.dataset.get_max_movie_id()+1,self.embedding_size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.1))
        self.user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user)
        self.pos_item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item_pos)
        self.neg_item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item_neg)

        #loss
        td=tf.reduce_sum(tf.multiply(self.user_embedding,(self.pos_item_embedding-self.neg_item_embedding)),axis=1,keepdims=True)
        l2_norm=tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding,self.user_embedding)),
            tf.reduce_sum(tf.multiply(self.pos_item_embedding,self.pos_item_embedding)),
            tf.reduce_sum(tf.multiply(self.neg_item_embedding,self.neg_item_embedding))
        ])
        self.loss=self.reg_mf*l2_norm-tf.reduce_mean(tf.log(tf.nn.sigmoid(td)))
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def logging(self):
        self.logger.info("------------BPR---------------")
        self.logger.info("learing_rate:"+str(self.lr))
        self.logger.info("reg_mf:"+str(self.reg_mf))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("data_number"+str(self.number))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("num_neg:" + str(self.num_neg))

    def train(self):
        """
        training and store the log of the training
        """
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger=Logger("D:/PycharmProjects/recommend_DIY/log/BPR/%s"%str(localtime))
        self.evaluate = Evaluate(dataset=self.dataset, number=self.number, logger=self.logger,model=self)
        self.logging()

        for episode in range(self.episodes):
            total_loss=0.0
            train_start_time=timee()
            data_iter=self.sampler.get_train_batch()
            num_batch=self.sampler.get_batch_number()
            for batch_i in range(num_batch):
                data=next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.user:data[:,0],
                    self.item_pos:data[:,1],
                    self.item_neg:data[:,2]
                })
                total_loss+=loss
            self.logger.info("--Episode %d:     loss: %.3f      running_time: %.3f"%(episode,total_loss/num_batch,timee()-train_start_time))
            if (episode+1)%self.verbose==0:
                self.evaluate.F1(top_k=20,batch_size=400)
                self.evaluate.F1(top_k=40, batch_size=400)
                self.evaluate.F1(top_k=60, batch_size=400)
                self.evaluate.F1(top_k=80, batch_size=400)
        """
        store the model
        """
        values = self.sess.run(self.var)
        pickle.dump((values), open('D:/PycharmProjects/recommend_DIY/dataset/bpr.p', 'wb'))

    def predict(self,user_ids,items=None):
        user_matrix,item_matrix=self.sess.run([self.user_matrix,self.item_matrix])
        user_matrix=user_matrix[user_ids]
        ratings=np.matmul(user_matrix,item_matrix.T)
        if items is not None:
            ratings=[rating[item] for rating,item in zip(ratings,items)]
        return ratings

sess=tf.Session()
bpr=BPR(sess=sess)
sess.run(tf.global_variables_initializer())
bpr.train()