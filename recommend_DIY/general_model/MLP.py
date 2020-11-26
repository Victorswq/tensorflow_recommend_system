import tensorflow as tf
import numpy as np
from time import time as timee
import time

from data.sampler import PointSampler
from data.dataset import Dataset
from evaluate.logger import Logger
from evaluate.evaluate import Evaluate

class MLP(object):
    def __init__(self,
                 sess,
                 learning_rate=0.001,
                 batch_size=512,
                 data_name="ml_100k",
                 embedding_size=64,
                 layers=[64],
                 episodes=1000,
                 verbose=10,
                 reg_mf=0.00001,
                 num_neg=1,
                 ):
        self.sess=sess
        self.batch_size=batch_size
        self.lr=learning_rate
        self.data_name=data_name
        self.embedding_size=embedding_size
        self.layers=layers
        self.episodes=episodes
        self.verbose=verbose
        self.reg_mf=reg_mf
        self.num_neg=num_neg

        self.sampler=PointSampler(batch_size=self.batch_size,data_name="ml_100k",num_neg=self.num_neg)
        self.dataset=Dataset(data_name=self.data_name)
        self.max_movie_id = self.dataset.get_max_movie_id()+1
        self.max_user_id=self.dataset.get_max_user_id()+1

        self.build_net()

    def build_net(self):
        #placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.label=tf.placeholder(tf.int32,[None,],name="label")

        #variables
        self.user_matrix=tf.get_variable("user_matrix",shape=[self.max_user_id,self.embedding_size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.1))
        print(self.max_movie_id)
        self.item_matrix=tf.get_variable("item_matrix",shape=[self.max_movie_id,self.embedding_size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.1))
        self.user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user)
        self.item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item)
        vecotor=tf.concat([self.user_embedding,self.item_embedding],axis=1)
        # vecotor=tf.reduce_mean(vecotor,axis=1,keep_dims=True)
        # vecotor=tf.multiply(self.user_embedding,self.item_embedding)

        #net
        l2_norm=tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding,self.user_embedding)),
            tf.reduce_sum(tf.multiply(self.item_embedding,self.item_embedding)),
        ])
        for idx in range(len(self.layers)):
            vecotor=tf.layers.dense(vecotor,self.layers[idx],activation=tf.nn.relu)
            vecotor=tf.layers.dropout(vecotor)
        self.inference=tf.reduce_sum(vecotor,axis=1)
        self.loss=tf.losses.mean_squared_error(labels=self.label,predictions=self.inference)
        self.loss+=self.reg_mf*l2_norm
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def logging(self):
        self.logger.info("--------------MLP-------------")
        self.logger.info("learing_rate:"+str(self.lr))
        self.logger.info("layers:"+str(self.layers))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("data_name"+str(self.data_name))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("reg_mf:" + str(self.reg_mf))
        self.logger.info("num_neg:" + str(self.num_neg))

    def train(self):
        saver = tf.train.Saver()
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/MLP/%s" % str(localtime))
        self.evaluate=Evaluate(dataset=self.dataset,logger=self.logger,model=self,data_name=self.data_name)
        self.logging()

        for episode in range(self.episodes):
            total_loss=0.0
            train_start_time=timee()
            data_iter=self.sampler.get_train_batch()
            batch_num=self.sampler.get_batch_number()
            for batch in range(batch_num):
                data=next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.user:data[:,0],
                    self.item:data[:,1],
                    self.label:data[:,2]
                })
                total_loss+=loss
            self.logger.info("--Episode %d:    loss: %.3f     time: %.3f"%(episode,total_loss/batch_num,timee()-train_start_time))
            if (episode+1)%self.verbose==0:
                self.evaluate.F1(top_k=[20,30,40],batch_size=400)
        saver.save(self.sess, 'D:/PycharmProjects/recommend_DIY/temp/model_MLP')
        print('Model Trained and Saved')


    def predict(self,user_ids,items=None):
        ratings=[]
        if items is not None:
            for user_id,item in zip(user_ids,items):
                user=np.full(len(item),user_id)
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:user,
                    self.item:item
                })
                ratings.append(rating)
        else:
            for user_id in user_ids:
                user=np.full(self.max_movie_id,user_id)
                item=np.arange(self.max_movie_id)
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:user,
                    self.item:item
                })
                ratings.append(rating)
        return ratings

sess=tf.Session()
mlp=MLP(sess)
sess.run(tf.global_variables_initializer())
mlp.train()