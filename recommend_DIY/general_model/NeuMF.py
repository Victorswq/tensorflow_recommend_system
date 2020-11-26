import tensorflow as tf
import numpy as np
from time import time as timee
import time

from data.sampler import PointSampler
from data.dataset import Dataset
from evaluate.evaluate import Evaluate
from evaluate.logger import Logger


class NeuMF(object):
    def __init__(self,
                 sess,
                 learning_rate=0.00001,
                 batch_size=512,
                 verbose=50,
                 episodes=1000,
                 embedding_size=32,
                 number="ml_1m",
                 layers=[32,32],
                 reg_mf=0.0001,
                 num_neg=0
                 ):
        self.sess=sess
        self.lr=learning_rate
        self.batch_size=batch_size
        self.verbose=verbose
        self.episodes=episodes
        self.embedding_size=embedding_size
        self.number=number
        self.layers=layers
        self.reg_mf=reg_mf
        self.num_neg=num_neg

        self.dataset=Dataset(data_name=self.number)
        self.sampler=PointSampler(batch_size=self.batch_size,number=self.number,num_neg=self.num_neg)
        self.max_movie_id = self.dataset.get_max_movie_id() + 1
        self.max_user_id = self.dataset.get_max_user_id() + 1
        self.build_net()

    def build_net(self):
        #placeholder
        self.user=tf.placeholder(tf.int32,shape=[None,],name="user")
        self.item=tf.placeholder(tf.int32,shape=[None,],name="item")
        self.label=tf.placeholder(tf.float32,shape=[None,],name="label")

        #variable
        self.mf_user_matrix=tf.get_variable("mf_user_matrix",
                                            shape=[self.max_user_id,self.embedding_size],
                                            dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(0.,.1))
        self.mf_item_matrix=tf.get_variable("mf_item_matrix",
                                            shape=[self.max_movie_id,self.embedding_size],
                                            dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(0.,.1))
        self.mlp_user_matrix=tf.get_variable("mlp_user_matrix",
                                             shape=[self.max_user_id,self.layers[0]],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(0.,.1))
        self.mlp_item_matrix=tf.get_variable("mlp_item_matrix",
                                             shape=[self.max_movie_id,self.layers[0]],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(0.,.1))

        mf_user_embedding=tf.nn.embedding_lookup(self.mf_user_matrix,self.user)
        mf_item_embedding=tf.nn.embedding_lookup(self.mf_item_matrix,self.item)
        mlp_user_embedding=tf.nn.embedding_lookup(self.mlp_user_matrix,self.user)
        mlp_item_embedding=tf.nn.embedding_lookup(self.mlp_item_matrix,self.item)

        l2_norm=tf.add_n([
            tf.reduce_sum(tf.multiply(mf_user_embedding,mf_user_embedding)),
            tf.reduce_sum(tf.multiply(mf_item_embedding,mf_item_embedding)),
            tf.reduce_sum(tf.multiply(mlp_user_embedding,mlp_user_embedding)),
            tf.reduce_sum(tf.multiply(mlp_item_embedding,mlp_item_embedding))
        ])

        mf_vector=tf.multiply(mf_user_embedding,mf_item_embedding)
        mlp_vector=tf.concat([mlp_user_embedding,mlp_item_embedding],axis=1)

        for idx in range(len(self.layers)):
            mlp_vector=tf.layers.dense(mlp_vector,
                                       self.layers[idx],
                                       activation=tf.nn.relu)

        self.inference=tf.reduce_sum(tf.concat([mf_vector,mlp_vector],axis=1),axis=1)

        #loss and optimizer
        self.loss=tf.losses.mean_squared_error(labels=self.label,predictions=self.inference)
        self.loss+=self.reg_mf*l2_norm
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def logging(self):
        self.logger.info("--------------NeuMF-------------")
        self.logger.info("learing_rate:" + str(self.lr))
        self.logger.info("layers:" + str(self.layers))
        self.logger.info("batch_size:" + str(self.batch_size))
        self.logger.info("embedding_size:" + str(self.embedding_size))
        self.logger.info("data_number" + str(self.number))
        self.logger.info("number_of_epochs:" + str(self.episodes))
        self.logger.info("verbose:" + str(self.verbose))
        self.logger.info("reg_mf:" + str(self.reg_mf))
        self.logger.info("num_neg:" + str(self.num_neg))

    def train(self):
        saver = tf.train.Saver()
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/NeuMF/%s" % str(localtime))
        self.evaluate = Evaluate(dataset=self.dataset, logger=self.logger, model=self, number=self.number)
        self.logging()

        for episode in range(self.episodes):
            total_loss=0.0
            train_start_time=timee()
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            for batch in range(batch_number):
                data=next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.user:data[:,0],
                    self.item:data[:,1],
                    self.label:data[:,2]
                })
                total_loss+=loss
            self.logger.info("--Episode %d:    loss: %.3f     time: %.3f"%(episode,total_loss/batch_number,timee()-train_start_time))
            if (episode+1)%self.verbose==0:
                self.evaluate.F1(top_k=20,batch_size=400)
                self.evaluate.F1(top_k=40, batch_size=400)
                self.evaluate.F1(top_k=60, batch_size=400)
                self.evaluate.F1(top_k=80, batch_size=400)
        saver.save(self.sess,"D:/PycharmProjects/recommend_DIY/temp/model_NeuMF")
        print("Model Trained and Saved")

    def predict(self,user_ids,items=None):
        ratings=[]
        if items is not None:
            for user_id,item in zip(user_ids,items):
                user=np.full(len(items),user_id)
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:user,
                    self.item:item
                })
                ratings.append(rating)
        else:
            for user_id in user_ids:
                user=np.full(self.max_movie_id_,user_id)
                item=np.arange(self.max_movie_id)
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:user,
                    self.item:item
                })
                ratings.append(rating)
        return ratings

sess=tf.Session()
neumf=NeuMF(sess=sess)
sess.run(tf.global_variables_initializer())
neumf.train()