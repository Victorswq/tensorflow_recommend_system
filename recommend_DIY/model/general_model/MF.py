import numpy as np
import tensorflow as tf

from model.AbstractRecommender import General_Model
from data.sampler import PointSampler,PairwiseSampler


class MF(General_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="MF",
                 learning_rate=0.001,
                 episodes=100,
                 batch_size=512,
                 verbose=5,
                 reg_rate=0.0001,
                 embedding_size=32,
                 is_pairwise=False,
                 loss_function="square"
                 ):
        super(MF,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.batch_size=batch_size
        self.verbose=verbose
        self.reg_rate=reg_rate
        self.embedding_size=embedding_size
        self.is_pairwise=is_pairwise
        self.loss_function=loss_function

        if self.is_pairwise is True:
            self.sampler=PairwiseSampler(batch_size=self.batch_size,data_name=self.data_name)
        else:
            self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name)
        self.build_model()

    def build_placeholder(self):
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        if self.is_pairwise is True:
            self.neg_item=tf.placeholder(tf.int32,[None,],name="neg_item")
        else:
            self.label=tf.placeholder(tf.float32,[None,],name="label")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.user_matrix=tf.get_variable(name="user_matrix",shape=[self.num_users,self.embedding_size],initializer=init_w)
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)

    def build_inference(self,item_input):
        user_embedding = tf.nn.embedding_lookup(self.user_matrix, self.user)
        item_embedding=tf.nn.embedding_lookup(self.item_matrix,item_input)
        inference=tf.reduce_sum(tf.multiply(user_embedding,item_embedding),axis=1)
        return user_embedding,item_embedding,inference

    def build_graph(self):
        self.user_embedding,self.item_embeedding,self.inference=self.build_inference(item_input=self.item)
        if self.is_pairwise is True:
            _,self.neg_item_embedding,neg_inference=self.build_inference(item_input=self.neg_item)
            self.loss=self.pairwise_loss(self.inference,neg_inference,loss_function=self.loss_function)+self.reg_rate*self.get_l2_loss(self.user_embedding,self.item_embeedding,self.neg_item_embedding)
        else:
            self.loss=self.pointwise_loss(label=self.label,prediction=self.inference,loss_function=self.loss_function)+self.reg_rate*self.get_l2_loss(self.user_embedding,self.item_embeedding)
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def train(self):
        for episode in range(1,1+self.episodes):
            training_start_time=self.get_time()
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            total_loss=0
            if self.is_pairwise is not True:
                for i in range(1,1+batch_number):
                    data=next(data_iter)
                    loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.label:data[:,2]
                    })
                    total_loss+=loss
            else:
                for i in range(1,1+batch_number):
                    data=next(data_iter)
                    loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.neg_item:data[:,2]
                    })
                    total_loss+=loss
            self.record_train(episode=episode,total_loss=total_loss,batch_number=batch_number,training_start_time=training_start_time)
            if episode%self.verbose==0:
                user_ids,ratings=self.evaluate.get_ratings(test_size=512)
                self.evaluate.F1(user_ids=user_ids,ratingss=ratings)
                self.evaluate.HitRate(user_ids=user_ids,ratingss=ratings,top_k=[1,2,5])
                self.evaluate.NDCG_plus(user_ids=user_ids,ratingss=ratings)
                self.evaluate.MRR(user_ids=user_ids, ratingss=ratings)

    def predict(self,user_ids,items=None):
        user_matrix,item_matrix=self.sess.run([self.user_matrix,self.item_matrix])
        user_matrix=user_matrix[user_ids]
        ratings=np.matmul(user_matrix,item_matrix.T)
        if items is not None:
            ratings=[rating[item] for rating,item in zip(ratings,items)]
        return ratings

sess=tf.Session()
mf=MF(sess=sess,is_pairwise=True,loss_function="bpr",verbose=5)
sess.run(tf.global_variables_initializer())
mf.train()