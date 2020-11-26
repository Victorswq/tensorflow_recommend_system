import numpy as np
import tensorflow as tf

from model.AbstractRecommender import General_Model


class NeuMF(General_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="NeuMF",
                 learning_rate=0.001,
                 is_pairwise=False,
                 batch_size=512,
                 episodes=100,
                 verbose=5,
                 embedding_size=32,
                 layers=[64,32],
                 loss_function="cross_entropy",
                 trainer="adam",
                 reg_rate=0.0001,
                 ):
        super(NeuMF,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.learning_rate=learning_rate
        self.is_pairwise=is_pairwise
        self.batch_size=batch_size
        self.episodes=episodes
        self.verbose=verbose
        self.embedding_size=embedding_size
        self.layers=layers
        self.loss_function=loss_function
        self.trainer=trainer
        self.reg_rate=reg_rate

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
        self.mf_user_matrix=tf.get_variable(name="mf_user_matrix",shape=[self.num_users,self.embedding_size],initializer=init_w)
        self.mf_item_matrix=tf.get_variable(name="mf_item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.mlp_user_matrix=tf.get_variable(name="mlp_user_matrix",shape=[self.num_users,self.layers[0]/2],initializer=init_w)
        self.mlp_item_matrix=tf.get_variable(name="mlp_item_matrix",shape=[self.num_items,self.layers[0]/2],initializer=init_w)
        self.mlp_layers=[tf.layers.Dense(layer,activation=tf.nn.relu,name="mlp_layer_%d"%idx,kernel_initializer=init_w,bias_initializer=init_b) for idx,layer in enumerate(self.layers)]

    def build_inference(self,item_input):
        mf_user_embedding=tf.nn.embedding_lookup(self.mf_user_matrix,self.user)
        mf_item_embedding=tf.nn.embedding_lookup(self.mf_item_matrix,item_input)
        mf_vector=tf.multiply(mf_user_embedding,mf_item_embedding)
        mlp_user_embedding=tf.nn.embedding_lookup(self.mlp_user_matrix,self.user)
        mlp_item_embedding=tf.nn.embedding_lookup(self.mlp_item_matrix,item_input)
        mlp_vector=tf.concat([mlp_user_embedding,mlp_item_embedding],axis=1)
        for layer in self.mlp_layers:
            mlp_vector=layer(mlp_vector)
        inference=tf.reduce_sum(tf.concat([mf_vector,mlp_vector],axis=1),axis=1)
        return mf_user_embedding,mf_item_embedding,mlp_user_embedding,mlp_item_embedding,inference

    def build_graph(self):
        mf_user_embedding,mf_item_embedding,mlp_user_embedding,mlp_item_embedding,self.inference=self.build_inference(self.item)
        if self.is_pairwise is True:
            _,mf_neg_item_embedding,_,mlp_neg_item_embedding,neg_output=self.build_inference(self.neg_item)
            self.loss=self.pairwise_loss(self.inference,neg_output,loss_function=self.loss_function)+\
                      self.reg_rate*self.get_l2_loss(mf_user_embedding,mf_item_embedding,mlp_user_embedding,mlp_item_embedding,mf_neg_item_embedding,mlp_neg_item_embedding)
        else:
            self.loss=self.pointwise_loss(self.label,self.inference,loss_function=self.loss_function)+self.reg_rate*self.get_l2_loss(mf_user_embedding,mf_item_embedding,mlp_user_embedding,mlp_item_embedding)
        self.train_op=self.optimizer(self.loss,trainer=self.trainer)

    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def train(self):
        for episode in range(1,1+self.episodes):
            training_start_time=self.get_time()
            total_loss=0
            data_iter=self.sampler.get_train_batch()
            batch_number=self.sampler.get_batch_number()
            for i in range(1,1+batch_number):
                data=next(data_iter)
                if self.is_pairwise is True:
                    feed_dict={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.neg_item:data[:,2]
                    }
                else:
                    feed_dict={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.label:data[:,2]
                    }
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                total_loss+=loss
            self.record_train(episode,total_loss,batch_number,training_start_time)
            if episode%self.verbose==0:
                user_ids,ratings=self.evaluate.get_ratings(test_size=self.batch_size)
                self.evaluate.F1(ratingss=ratings,user_ids=user_ids)
                self.evaluate.HitRate(user_ids=user_ids,ratingss=ratings)

    def predict(self,user_ids,items=None):
        ratings=[]
        for user_id in user_ids:
            user=np.full(self.num_items,user_id)
            item=np.arange(self.num_items)
            rating=self.sess.run(self.inference,feed_dict={
                self.user:user,
                self.item:item
            })
            ratings.append(rating)
        if items is not None:
            ratings=[rating[item] for rating,item in zip(ratings,items)]
        return ratings

sess=tf.Session()
NeuNF=NeuMF(sess=sess,is_pairwise=True,loss_function="bpr")
sess.run(tf.global_variables_initializer())
NeuNF.train()