import numpy as np
import tensorflow as tf

from model.AbstractRecommender import General_Model


class NFM(General_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="NFM",
                 learning_rate=0.001,
                 batch_size=512,
                 verbose=5,
                 episodes=100,
                 embedding_size=32,
                 feature_size=None,
                 feature_length=None,
                 layers=[64,32],
                 loss_function="square",
                 trainer="adam",
                 ):
        super(NFM,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.verbose=verbose
        self.episodes=episodes
        self.embedding_size=embedding_size
        self.feature_length=feature_length
        self.layers=layers
        self.loss_function=loss_function
        self.trainer=trainer

        self.feature_size=feature_size

    def build_placeholder(self):
        self.feature=tf.placeholder(tf.int32,[None,self.feature_size],name="feature")
        self.label=tf.placeholder(tf.float32,[None,1],name="label")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.FM_second_matrix=tf.get_variable(name="FM_seconde_matrix",shape=[self.feature_length,self.embedding_size],initializer=init_w)
        self.FM_first_matrix=tf.get_variable(name="FM_first_matrix",shape=[self.feature_length,1],initializer=init_w)
        self.NFM_layer=[tf.layers.Dense(layer,activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name="layer_%d"%idx) for idx,layer in enumerate(self.layers)]
        self.prediction_layer=tf.layers.Dense(1,activation=tf.nn.sigmoid,kernel_initializer=init_w,bias_initializer=init_b,name="prediction_layer")

    def build_inference(self,item_input):
        FM_second_embedding=tf.nn.embedding_lookup(self.FM_second_matrix,item_input) # N * feature_size * embedding
        FM_first_embedding=tf.nn.embedding_lookup(self.FM_first_matrix,item_input) # N * feature_size * 1
        sum_square=tf.reduce_sum(FM_second_embedding,axis=1) # N * embedding
        sum_square=tf.square(sum_square) # N * embedding
        square_sum=tf.square(FM_second_embedding) # N * feature_size * embedding
        square_sum=tf.reduce_sum(square_sum,axis=1) # N * embedding
        FM_second=0.5*(tf.subtract(sum_square,square_sum)) # N * embedding
        for layer in self.NFM_layer:
            FM_second=layer(FM_second)
        # N * final_layer_size
        FM_second=self.prediction_layer(FM_second) # N * 1
        FM_first=tf.squeeze(FM_first_embedding) # N * feature_size
        FM_first=tf.reduce_sum(FM_first,axis=1) # N * 1
        inference=tf.add(FM_first_embedding,FM_first) # N * 1
        return FM_first_embedding,FM_second_embedding,inference

    def build_graph(self):
        FM_first_embedding,FM_second_embedding,self.inference=self.build_inference(item_input=self.feature)
        self.loss=self.pointwise_loss(self.label,self.inference,loss_function=self.loss_function)+self.get_l2_loss(FM_first_embedding,FM_second_embedding)
        self.train_op=self.optimizer(self.loss,self.trainer)

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
            for i in range(1,1+batch_number):
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.feature:data_iter[0,:3],
                    self.label:data_iter[:,4:-1]
                })
                total_loss+=loss
            self.record_train(episode,total_loss,batch_number,training_start_time)
            if episode%self.verbose==0:
                self.evaluate.HR()

    def predict(self,user_ids,items=None):
        pass