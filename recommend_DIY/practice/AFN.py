import tensorflow as tf
import numpy as np

from time import time
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import roc_auc_score


class AFM(BaseEstimator,TransformerMixin):
    def __init__(self,
                 feature_size,
                 field_size,
                 embedding_size=8,
                 attention_size=10,
                 deep_layers=[32,32],
                 deep_init_size=50,
                 dropout_deep=[0.5,0.5,0.5],
                 dropout_layer_activation=tf.nn.relu,
                 epoch=10,
                 batch_size=256,
                 learning_rate=0.001,
                 batch_norm=0,
                 batch_norm_decay=0.995,
                 verbose=False,
                 random_seed=2016,
                 eval_metric=roc_auc_score,
                 gerater_is_better=True,
                 user_inner=True
                 ):
        self.feature_size=feature_size
        self.field_size=field_size
        self.embedding_size=embedding_size
        self.attention_size=attention_size

        self.deep_layers=deep_layers
        self.deep_init_size=deep_init_size
        self.dropout_dep=dropout_deep
        self.deep_layers_activation=dropout_layer_activation

        self.epoch=epoch
        self.batch_size=batch_size
        self.learning_rate=learning_rate

        self.batch_norm=batch_norm
        self.batch_norm_decay=batch_norm_decay

        self.verbose=verbose
        self.random_seed=random_seed
        self.eval_metric=eval_metric
        self.greater_is_better=gerater_is_better
        self.train_result,self.valid_result=[],[]

        self.user_inner=user_inner

    def initialize_weights(self):
        weights=dict()
        init=tf.random_normal_initializer(0.,.1)
        #embeddings
        weights["feature_embeddings"]=tf.get_variable(
            name="feature_embeddings",
            shape=[self.feature_size,self.eval_metric],
            initializer=init,
        )
        weights["bias"]=tf.Variable(tf.constant(0.1),name="bias")

        return weights

    def init_graph(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feat_index=tf.placeholder(
                tf.int32,
                shape=[None,None],
                name="feat_index"
            )
            self.feat_value=tf.placeholder(
                tf.float32,
                shape=[None,None],
                name="feat_value"
            )
            self.label=tf.placeholder(tf.float32,[None,1],name="label")
            self.dropout_keep_deep=tf.placeholder(
                tf.float32,shape=[None,1],
                name="dropout_deep_deep"
            )
            self.train_phase=tf.placeholder(tf.bool,name="train_phase")
            self.weight=self.initialize_weights()

            #embedding
            self.embeddings=tf.nn.embedding_lookup(self.weight["feature_embeddings"],self.feat_index)
            feat_value=tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            self.embeddings=tf.multiply(self.embeddings,feat_value)

            #element_wise
            element_wise_product_list=[]
            for i in range(self.field_size):
                for j in range(i+1,self.field_size):
                    element_wise_product_list.append(tf.multiply(self.embeddings[:,i,:],self.embeddings[:,j,:]))

            self.element_wise_product=tf.stack(element_wise_product_list)
            self.element_wise_product=tf.transpose(self.element_wise_product,
                                                   perm=[1,0,2],
                                                   name="element_wise_product")

            #attention part
            self.attention_part=tf.reshape(self.element_wise_product,shape=(-1,self.embedding_size))
            num_interactions=int(self.field_size*(self.field_size-1)/2)
            attention=tf.layers.dense(
                inputs=self.attention_part,
                units=self.attention_size,
                activation=tf.nn.relu,
            )
            attention=tf.reshape(attention,shape=(-1,num_interactions,self.attention_size))
            attention=tf.reduce_sum(attention,axis=1,keepdims=True)
