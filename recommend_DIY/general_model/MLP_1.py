import numpy as np
import tensorflow as tf

from general_model.AbstractRecommender import AbstractRecommender
from data.sampler import PointSampler,PairwiseSampler


class MLP(AbstractRecommender):
    def __init__(self,
                 sess,
                 string="MLP_1",
                 data_name="ml_100k",
                 learning_rate=0.001,
                 batch_size=512,
                 episodes=100,
                 verbose=5,
                 embedding_size=32,
                 layers=[32,1],
                 is_pairwise=False,
                 num_neg=1,
                 ):
        super(MLP,self).__init__(data_name=data_name,string=string,layers=layers)
        self.sess=sess
        self.string=string
        self.data_name=data_name
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.episodes=episodes
        self.verbose=verbose
        self.embedding_size=embedding_size
        self.layers=layers
        self.is_pairwise=is_pairwise
        self.num_neg=num_neg

        if self.is_pairwise is True:
            self.sampler=PairwiseSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=1)
        else:
            self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=self.num_neg)
        self.build_graph()

    def build_graph(self):
        self.build_tools()
        self.build_net()

    def build_net(self):
        # placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.label=tf.placeholder(tf.float32,[None,],name="label")
        self.neg_item=tf.placeholder(tf.int32,[None,],name="neg_item") # if loss is BPR

        # matrix and embedding
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.user_matrix=tf.get_variable(name="user_matrix",shape=[self.num_users,self.embedding_size],initializer=init_w)
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user)
        self.item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.item)

        # mlp
        if self.is_pairwise is True:
            self.neg_item_embedding=tf.nn.embedding_lookup(self.item_matrix,self.neg_item)
            self.vector_1=tf.concat([self.user_embedding,self.neg_item_embedding],axis=1)
            self.vector_2=tf.concat([self.user_embedding,self.item_embedding],axis=1)
            self.vector=tf.concat([self.vector_1,self.vector_2],axis=0)
        else:
            self.vector=tf.concat([self.user_embedding,self.item_embedding],axis=1)
        for i in range(len(self.layers)):
            self.verctor=tf.layers.dense(self.vector,self.layers[i],activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name="layer_%d"%i)
        self.inference=tf.reduce_mean(self.vector,axis=1)
        if self.is_pairwise is True:
            half=self.batch_size
            self.loss=tf.reduce_mean(tf.log_sigmoid(self.inference[:half]-self.inference[half:]))+self.reg_mf*self.l2_loss(self.user_embedding,self.item_embedding,self.neg_item_embedding)
        else:
            self.loss=tf.reduce_mean(tf.square(self.inference-self.label))+self.reg_mf*self.l2_loss(self.user_embedding,self.item_embedding)
        # train and optimizer
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        for episode in range(1,1+self.episodes):
            total_loss=0.
            train_start_time=self.get_time()
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            for i in range(batch_number-1):
                data=next(data_iter)
                if self.is_pairwise is True:
                    feed={self.user:data[:,0],
                          self.item:data[:,1],
                          self.neg_item:data[:,2]}
                else:
                    feed={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.label:data[:,2]
                    }
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed)
                total_loss+=loss
            self.logger.info("Episode %d:    loss: %.3f     running_time: %.3f"%(episode,total_loss/batch_number,self.get_time()-train_start_time))
            if episode%self.verbose==0:
                self.evaluate.F1(batch_size=self.batch_size)

    def predict(self,user_ids,item_ids=None):
        ratings=[]
        if item_ids is None:
            for user_id in user_ids:
                user=np.full(self.num_items,user_id)
                item=np.arange(self.num_items)
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:user,
                    self.item:item
                })
                ratings.append(rating)
        return ratings
sess=tf.Session()
mlp=MLP(sess=sess,is_pairwise=False)
sess.run(tf.global_variables_initializer())
mlp.train()