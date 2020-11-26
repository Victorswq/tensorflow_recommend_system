import numpy as np
import tensorflow as tf

from general_model.AbstractRecommender import AbstractRecommender

class MF(AbstractRecommender):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="MF",
                 learning_rate=0.001,
                 batch_size=512,
                 verbose=5,
                 episodes=100,
                 is_pairwise=False,

                 ):
        super(MF,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.verbose=verbose
        self.episodes=episodes
        self.is_pairwise=is_pairwise


    def build_net(self):
        # placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        if self.is_pairwise is not True:
            self.label=tf.placeholder(tf.float32,[None,],name="label")
        else:
            self.neg_item=tf.placeholder(tf.int32,[None,],name="neg_item")
