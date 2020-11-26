import numpy as np
import tensorflow as tf

from model.AbstractRecommender import General_Model


class MLP(General_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="MLP",
                 learning_rate=0.001,
                 batch_size=512,
                 episodes=100,
                 embedding_size=32,
                 verbose=5,
                 is_pairwise=False,
                 layers=[32,16],
                 loss_function="square",
                 reg_rate=0.1,
                 trainer="adam",
                 is_store_model=False,
                 num_neg=1,
                 ):
        super(MLP,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.episodes=episodes
        self.embedding_size=embedding_size
        self.verbose=verbose
        self.is_pairwise=is_pairwise
        self.layers=layers
        self.loss_function=loss_function
        self.reg_rate=reg_rate
        self.trainer=trainer
        self.is_store_model=is_store_model
        self.num_neg=num_neg

        self.build_model()

    def build_placeholder(self):
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        if self.is_pairwise is True:
            self.neg_item=tf.placeholder(tf.int32,[None,],name="neg_item")
        else:
            self.label=tf.placeholder(tf.float32,[None,],name="label")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.1)
        init_b=tf.constant_initializer(0.)
        self.user_matrix=tf.get_variable(name="user_matrix",shape=[self.num_users,self.layers[0]/2],initializer=init_w)
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.num_items,self.layers[0]/2],initializer=init_w)
        for i in range(len(self.layers)):
            self.mlp_layers=[tf.layers.Dense(self.layers[i],activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name="layer_%d"%(i+1))]

    def build_inference(self,item_input):
        user_embedding=tf.nn.embedding_lookup(self.user_matrix,self.user) # N * embedding_size
        item_embedding=tf.nn.embedding_lookup(self.item_matrix,item_input) # N * embedding_size
        vector=tf.concat([user_embedding,item_embedding],axis=1) # N * 2_mul_embedding_size
        for layer in self.mlp_layers:
            vector=layer(vector)
        vector=tf.reduce_sum(vector,axis=1)
        return user_embedding,item_embedding,vector

    def build_graph(self):
        self.user_embedding,self.item_embedding,self.inference=self.build_inference(item_input=self.item)
        if self.is_pairwise is True:
            _,neg_item_embedding,neg_inference=self.build_inference(self.neg_item)
            self.loss=self.pairwise_loss(positive_inference=self.inference,negative_inference=neg_inference,loss_function=self.loss_function)+\
                      self.reg_rate*self.get_l2_loss(self.user_embedding,self.item_embedding,neg_item_embedding)
        else:
            # self.inference=tf.reduce_sum(self.inference,axis=1)
            self.loss=self.pointwise_loss(label=self.label,prediction=self.inference,loss_function=self.loss_function)+self.reg_rate*self.get_l2_loss(self.user_embedding,self.item_embedding)
        self.train_op=self.optimizer(self.loss,self.trainer)

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
                    feed = {
                        self.user:data[:, 0],
                        self.item:data[:,1],
                        self.neg_item:data[:,2]
                    }
                else:
                    feed={
                        self.user:data[:,0],
                        self.item:data[:,1],
                        self.label:data[:,2]
                    }
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed)
                total_loss+=loss
            self.record_train(episode=episode,total_loss=total_loss,batch_number=batch_number,training_start_time=training_start_time)
            if episode%self.verbose==0:
                user_ids,ratings=self.evaluate.get_ratings(test_size=self.batch_size)
                self.evaluate.F1(user_ids=user_ids,ratingss=ratings)
                self.evaluate.HitRate(user_ids=user_ids,ratingss=ratings)

    def predict(self,user_ids,items=None,):
        ratings=[]
        for user_id in user_ids:
            user=np.full(self.num_items,user_id)
            item=np.arange(self.num_items)
            rating=self.sess.run(self.inference,feed_dict={
                self.user:user,
                self.item:item,
            })
            ratings.append(rating)
        if items is not None:
            ratings=[rating[item] for rating,item in zip(ratings,items)]
        return ratings

sess=tf.Session()
mlp=MLP(sess=sess,learning_rate=0.0001,reg_rate=0.01,is_pairwise=True,loss_function="bpr")
sess.run(tf.global_variables_initializer())
mlp.train()