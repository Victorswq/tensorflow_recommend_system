import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model


class FPMC_plus(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="FPMC_plus",
                 high_order=5,):
        super(FPMC_plus, self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.high_order=high_order

        self.build_model()

    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def build_placeholder(self):
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.item_recent=tf.placeholder(tf.int32,[None,self.high_order],name="item_recent")
        if self.is_pairwise is True:
            self.item_neg=tf.placeholder(tf.int32,[None,],name="item_neg")
        else:
            self.label=tf.placeholder(tf.float32,[None,],name="label")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.user_matrix=tf.get_variable(name="user_matrix",shape=[self.num_users,self.embedding_size],initializer=init_w)
        self.item_matrix_for_user=tf.get_variable(name="item_matrix_for_user",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.item_matrix_for_recent_item=tf.get_variable(name="item_matrix_for_recent_item",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.item_recent_matrix=tf.get_variable(name="item_recent_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.attention_layer1=tf.layers.Dense(self.embedding_size,activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name="attention_layer1")
        self.attention_layer2=tf.layers.Dense(1,activation=None,kernel_initializer=init_w,bias_initializer=init_b,name="attention_layer2")

    def attention(self,user_embedding,item_embedding_for_recent_item,item_recent_embedding):
        # N * embedding_size ===>>> N * high_order * embedding_size
        user_embedding=tf.tile(tf.expand_dims(user_embedding,axis=1),tf.stack([1,self.high_order,1]))
        # N * embedding_size ===>>> N * high_order * embedding_size
        item_embedding_for_recent_item=tf.tile(tf.expand_dims(item_embedding_for_recent_item,axis=1),tf.stack([1,self.high_order,1]))
        # N * high_order * 3*embedding_size
        attention=tf.concat([user_embedding,item_embedding_for_recent_item,item_recent_embedding],axis=2)
        # N_mul_high_order * embedding_size
        attention=self.attention_layer1(tf.reshape(attention,[-1,3*self.embedding_size]))
        # N * high_order
        attention=tf.reshape(self.attention_layer2(attention),[-1,self.high_order])
        # N * high_order * 1
        attention=tf.expand_dims(tf.nn.softmax(attention,axis=1),2)
        # 把所有的最近item按照一定权重相加起来 N * embedding_size
        return tf.reduce_sum(tf.multiply(item_recent_embedding,attention),axis=1)

    def build_inference(self,item_input):
        user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user) # N * embedding_size
        item_embedding_for_user=tf.nn.embedding_lookup(self.item_matrix_for_user,item_input) # N * embedding_size
        item_embedding_for_recent_item=tf.nn.embedding_lookup(self.item_matrix_for_recent_item,item_input) # N * embedding_size
        item_recent_embedding=tf.nn.embedding_lookup(self.item_recent,self.item_recent) # N * high_order * embedding_size
        weight=self.attention(user_embedding,item_embedding_for_recent_item,item_recent_embedding) # N * embedding_size
        inference=tf.multiply(user_embedding,item_embedding_for_user)+tf.multiply(item_embedding_for_recent_item,weight) # N * embedding_size
        inference=tf.reduce_sum(inference,axis=1) # N,
        return user_embedding,item_embedding_for_user,item_embedding_for_recent_item,item_recent_embedding,inference

    def build_graph(self):
        user_embedding, item_embedding_for_user, item_embedding_for_recent_item, item_recent_embedding, self.inference=self.build_inference(self.item)
        if self.is_pairwise is True:
            _,neg_item_for_user,neg_item_for_recent_item,_,neg_output=self.build_inference(self.item_neg)
            l2_loss=self.get_l2_loss(user_embedding,item_embedding_for_user,item_embedding_for_recent_item,item_recent_embedding,neg_item_for_recent_item,neg_item_for_user)
            self.loss=self.pairwise_loss(self.inference,neg_output,loss_function=self.loss_function)+self.reg_rate*l2_loss
        else:
            l2_loss=self.get_l2_loss(user_embedding,item_embedding_for_user,item_embedding_for_recent_item,item_recent_embedding)
            self.loss=self.pointwise_loss(self.label,self.inference,loss_function=self.loss_function)+self.reg_rate*l2_loss
        self.train_op=self.optimizer(self.loss,trainer=self.trainer)

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
                        self.item_recent:data[:,2:2+self.high_order],
                        self.item_neg:data[:,-2]
                    }
                else:
                    feed={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.item_recent:data[:,2:2+self.high_order],
                        self.label:data[:,-1]
                    }
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed)
                total_loss+=loss
            self.record_train(episode,total_loss,batch_number,training_start_time)
            if episode % self.batch_size==0:
                user_ids,ratings=self.evaluate.get_ratings(test_size=self.batch_size)
                self.evaluate.F1(ratingss=ratings,user_ids=user_ids)

    def predict(self,user_ids,items=None):
        pass