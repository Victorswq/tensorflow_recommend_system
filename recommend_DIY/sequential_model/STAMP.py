import numpy as np
import tensorflow as tf
from time import time as timee
import time

from sequential_model.AbstractRecommender import AbstractRecommender
from data.dataset import Dataset
from evaluate.evaluate import Evaluate
from evaluate.logger import Logger
from data.sampler import SamplerSeq

class STAMP(AbstractRecommender):
    def __init__(self,
                 learning_rate=0.001,
                 episodes=100,
                 verbose=5,
                 embedding_size=64,
                 data_name="ml_100k",
                 batch_size=512,
                 ):
        super(STAMP,self).__init__()
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.verbose=verbose
        self.embedding_size=embedding_size
        self.data_name=data_name
        self.batch_size=batch_size

        self.dataset=Dataset(data_name=self.data_name)
        self.max_len = self.dataset.get_max_len()-1
        self.sampler=SamplerSeq(batch_size=self.batch_size,data_name=self.data_name)
        self.max_item_id=self.dataset.get_max_movie_id()+1

        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())


    def build_net(self):
        # placeholder
        self.item_seq=tf.placeholder(tf.int32,shape=[None,self.max_len],name="item_seq")
        self.label=tf.placeholder(tf.int32,shape=[None,],name="pos_seq")
        self.pos_last=tf.placeholder(tf.int32,shape=[None,1],name="item_last_click")

        # matrix
        init_w=tf.random_normal_initializer(0.,0.05)
        init_b=tf.constant_initializer(0.)
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.max_item_id,self.embedding_size],initializer=init_w,trainable=True)
        self.w0=tf.get_variable(name="w0",shape=[self.embedding_size,1],initializer=init_w,trainable=True)
        self.w1=tf.get_variable(name="w1",shape=[self.embedding_size,self.embedding_size],initializer=init_w,trainable=True)    # embedding_size * attention_size
        self.w2=tf.get_variable(name="w2",shape=[self.embedding_size,self.embedding_size],initializer=init_w,trainable=True)    # embedding_size * attention_size
        self.w3=tf.get_variable(name="w3",shape=[self.embedding_size,self.embedding_size],initializer=init_w,trainable=True)    # embedding_size * attention_size
        self.ba=tf.get_variable(name="ba",shape=[1,self.embedding_size],initializer=init_b,trainable=True)

        # embedding
        self.xi=tf.nn.embedding_lookup(self.item_matrix,self.item_seq)  # N * max_len * embedding_size
        self.ms=tf.reduce_mean(self.xi,axis=1)   # N * embedding_size
        self.xt=tf.reduce_mean(tf.nn.embedding_lookup(self.item_matrix,self.pos_last),axis=1)  # N * embedding_size
        # ai
        self.ai=tf.matmul(tf.reduce_sum(self.xi,axis=1),self.w1) # N * embedding_size
        self.ai+=tf.matmul(self.xt,self.w2) + tf.matmul(self.ms,self.w3) + self.ba # N * embedding_size
        self.ai=tf.nn.sigmoid(self.ai) # activation
        self.ai=tf.matmul(self.ai,self.w0) # N * 1

        # ma
        self.ma=tf.reduce_mean(tf.multiply(self.xi,self.ai[:,tf.newaxis]),axis=1) # N * embedding_size
        # hs
        self.hs=tf.layers.dense(self.ma,self.embedding_size,activation=tf.nn.tanh,kernel_initializer=init_w,bias_initializer=init_b,name="hs") # N * embedding_size
        # ht
        self.ht=tf.layers.dense(self.xt,self.embedding_size,activation=tf.nn.tanh,kernel_initializer=init_w,bias_initializer=init_b,name="ht") # N * embedding_size

        # y
        # item_matrix: num_items * embedding_size ===>>> N * num_items * embedding_size
        self.y=tf.tile(tf.expand_dims(self.item_matrix,axis=0),[tf.shape(self.hs)[0],1,1])
        # ht: N * embedding_size ===>>> N * 1 * embedding_size
        self.ht=tf.expand_dims(self.ht,axis=1)
        self.y=tf.multiply(self.ht,self.y) # N * num_items * embedding_size
        self.y=tf.matmul(tf.expand_dims(self.hs,axis=1),tf.transpose(self.y,[0,2,1])) # N * num_items
        self.y=tf.squeeze(self.y)

        # cross_entropy
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.y))

        # train_op
        self.train_op=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        for episode in range(self.episodes):
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            total_loss=0
            for i in range(batch_number):
                data=next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.item_seq:data[:,:-1],
                    self.label:data[:,-1],
                    self.pos_last:data[:,-2:-1]
                })
                print(loss)
                total_loss+=loss
            self.logger.info("Loss is %.3f"%(total_loss/batch_number))

    def build_tools(self):
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/STAMP/%s" % str(localtime))
        self.evaluate = Evaluate(logger=self.logger, model=self, data_name=self.data_name)
        self.evaluate.logging()
        self.logger.info("shuffle: %s" % self.max_len)

    def predict(self,user_ids,item_ids):
        pass

stamp=STAMP(data_name="diginetica",batch_size=8,learning_rate=0.01)
stamp.train()