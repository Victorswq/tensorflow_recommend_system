import tensorflow as tf
import numpy as np

from data.dataset import Dataset
from general_model.AbstractRecommender import AbstractRecommender

class AFM(AbstractRecommender):
    def __init__(self,
                 sess,
                 learning_rate=0.001,
                 batch_size=512,
                 episodes=100,
                 embedding_size=32,
                 layers=[32,32],
                 data_name="ml_100k",
                 dropout_layers=[0.5,0.5],
                 ):
        super(self,AFM).__init__()
        self.sess=sess
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.episodes=episodes
        self.embedding_size=embedding_size
        self.layers=layers
        self.data_name=data_name
        self.dropout_layers=dropout_layers

        self.dataset=Dataset(data_name=self.data_name)
        self.feature_size=self.dataset.get_feature_size()
        self.feature_max_len=self.dataset.get_feature_max_len()


    def build_tools(self):
        pass

    def build_net(self):
        # placeholder
        self.feature_index=tf.placeholder(tf.int32,[None,self.feature_size],name="feature_index")
        self.feature_value=tf.placeholder(tf.int32,[None,self.feature_size],name="feature_value")
        self.label=tf.placeholder(tf.int32,[None,],name="label")

        # matrix
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.feature_matrix=tf.get_variable(name="feature_matrix",shape=[self.feature_max_len,self.embedding_size],initializer=init_w)
        self.feature_matrix_first_order=tf.get_variable(name="feature_matrix_first_order",shape=[self.feature_max_len,1],initializer=init_w)

        # embedding and variables
        self.feature_embedding=tf.nn.embedding_lookup(self.feature_matrix,self.feature_index) # N * feature_size * embedding_size
        self.feature_value=self.feature_value[tf.newaxis,-1] # N * feature_size * 1
        self.feature_embedding=tf.multiply(self.feature_embedding,self.feature_value) # N * feature_size *embedding_size

        self.feature_embedding_first_order=tf.nn.embedding_lookup(self.feature_embedding_first_order,self.feature_index) # N * feature_size * 1
        self.feature_embedding_first_order=tf.squeeze(self.feature_embedding_first_order)
        self.feature_embedding_first_order=tf.multiply(self.feature_embedding_first_order,self.feature_value) # N * feature_size
        self.feature_embedding_first_order=self.feature_embedding_first_order[tf.newaxis,-1] # N * feature_size * 1

        self.bias=tf.get_variable(name="bias",shape=[1,self.feature_size],initializer=init_b) # N * feature_size
        self.bias=self.bias[tf.newaxis,-1] # N * feature_size * 1

        # FM
        feature_product=[]
        for i in range(self.feature_size):
            for j in range(i+1):
                feature_product.append(tf.multiply(self.feature_embedding[:,i,:],self.feature_embedding[:,j,:])) # feature_i * feature_j N * embedding_size

        feautures=tf.stack(feature_product)
        feautures=tf.transpose(feautures,[1,0,2])
        feautures=tf.reshape(feautures,[-1,self.embedding_size])

        # error
        """
        which is better isn't test, so this error maybe better than the model in the paper
        """
        self.sum_square=tf.reduce_sum(self.feature_embedding,axis=1) # N * embedding_size
        self.sum_square=tf.square(self.sum_square) # N * embedding_size
        self.square_sum=tf.square(self.feature_embedding) # N * feature_size * embedding_size
        self.square_sum=tf.reduce_sum(self.square_sum,axis=1) # N * embedding_size
        self.FM_second=0.5*(tf.subtract(self.sum_square,self.square_sum,name="sum_square_subtract_square_sum")) # N * embedding_size
        self.FM_second=tf.layers.dropout(self.FM_second,rate=self.dropout_layers[0],name='dropout_layer_input_FM_for_weight')

        # AFM_net
        self.layers[-1]=self.feature_size
        self.feature_weight=self.FM_second
        for i in range(len(self.layers)):
            self.feature_weight=tf.layers.dense(self.feature_weight,self.layers[i],activation=tf.nn.relu,name="attention_layer_%d"%i) # N * feature_size
            self.feature_weight=tf.layers.dropout(self.feature_weight,rate=self.dropout_layers[i],name="dropout_layer_%d"%i)
        self.feature_weight=self.feature_weight[tf.newaxis,-1]
        self.FM_with_weight=tf.multiply(self.feature_weight,self.feature_embedding) # N * feature_size * embedding_size
        self.embedding_for_prediction=self.FM_with_weight + self.bias + self.feature_embedding_first_order # N * feature_size * embedding_size
        self.embedding_for_prediction=tf.reduce_sum(self.embedding_for_prediction,axis=1) # N * embedding_size

        # prediction
        self.inference=tf.layers.dense(self.embedding_for_prediction,1,activation=tf.nn.sigmoid,name="prediction") # N * 1
        self.inference=tf.layers.dropout(self.inference,rate=self.dropout_layers[-1],name="dropout_for_prediction")

        # loss and train
        self.loss=tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label,self.inference))
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def predict(self,user_ids,item_ids):
        pass