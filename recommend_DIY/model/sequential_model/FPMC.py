import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model


class FPMC(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="FPMC",
                 ):
        super(FPMC, self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string

        self.build_model()


    def build_placeholder(self):
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.prev_item=tf.placeholder(tf.int32,[None,],name="prev_item")
        if self.is_pairwise is True:
            self.neg_item=tf.placeholder(tf.int32,[None,],name="neg_item")
        else:
            self.label=tf.placeholder(tf.float32,[None,],name="label")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.user_matrix=tf.get_variable(name="user_matrix",shape=[self.num_users,self.embedding_size],initializer=init_w)
        self.item_matrix_for_user=tf.get_variable(name="item_matrix_for_user",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.item_matrix_for_prev_item=tf.get_variable(name="item_matrix_for_prev_item",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.prev_item_matrix=tf.get_variable(name="prev_item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)

    def build_inference(self,item_input):
        user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user)
        item_embedding_for_user=tf.nn.embedding_lookup(self.item_matrix_for_user,item_input)
        user_item=tf.multiply(user_embedding,item_embedding_for_user)
        item_embedding_for_prev_item=tf.nn.embedding_lookup(self.item_matrix_for_prev_item,item_input)
        prev_item_embedding=tf.nn.embedding_lookup(self.prev_item_matrix,self.prev_item)
        item_prev_item=tf.multiply(item_embedding_for_prev_item,prev_item_embedding)
        inference=tf.reduce_sum(tf.add(user_item,item_prev_item),axis=1)
        return user_embedding,item_embedding_for_user,item_embedding_for_prev_item,prev_item_embedding,inference

    def build_graph(self):
        user_embedding, item_embedding_for_user, item_embedding_for_prev_item, prev_item_embedding, self.inference=self.build_inference(self.item)
        if self.is_pairwise is True:
            _,neg_item_for_user,neg_item_for_prev_item,_,neg_output=self.build_inference(self.neg_item)
            l2_loss=self.get_l2_loss(user_embedding,item_embedding_for_user,item_embedding_for_prev_item,prev_item_embedding,neg_item_for_user,neg_item_for_prev_item)
            self.loss=self.pairwise_loss(self.inference,neg_output,loss_function=self.loss_function)+self.reg_rate*l2_loss
        else:
            self.loss=self.pointwise_loss(self.label,self.inference,loss_function=self.loss_function)+self.reg_rate*self.get_l2_loss(user_embedding,item_embedding_for_user,item_embedding_for_prev_item,prev_item_embedding)
        self.train_op=self.optimizer(self.loss,trainer=self.trainer)

    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def train(self):
        for episode in range(1,1+self.episodes):
            total_loss=0
            training_start_time=self.get_time()
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            for i in range(batch_number):
                data=next(data_iter)
                if self.is_pairwise is True:
                    feed={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.prev_item:data[:,2],
                        self.neg_item:data[:,3]
                    }
                else:
                    feed={
                        self.user:data[:,0],
                        self.prev_item:data[:,2],
                        self.item:data[:,1],
                        self.label:data[:,3]
                    }
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed)
                total_loss+=loss
            self.record_train(episode,total_loss,batch_number,training_start_time)
            if episode%self.verbose==0:
                user_ids,ratings=self.evaluate.get_ratings(test_size=self.batch_size)
                self.evaluate.F1(ratingss=ratings,user_ids=user_ids)

    def predict(self,user_ids,items=None):
        pass