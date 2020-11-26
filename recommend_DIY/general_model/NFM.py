import tensorflow as tf
import numpy as np
from time import time as timee
import time
import pickle

from data.dataset import Dataset
from data.sampler import SamplerFM
from evaluate.evaluate import EvaluateFM
from evaluate.logger import Logger
from general_model.AbstractRecommender import AbstractRecommender

class NFM(AbstractRecommender):
    def __init__(self,
                 sess,
                 learning_rate=0.001,
                 batch_size=512,
                 embedding_size=64,
                 episodes=50,
                 data_name="frappe",
                 field_size=2,
                 layers=[64,64,32],
                 reg_mf=0.000001,
                 num_neg=1,
                 verbose=5,
                 pre_train=False,
                 dropout_rate=1,
                 shuffle=True,
                 user_layers=True,
                 ):
        super(NFM,self).__init__()
        self.sess=sess
        self.lr=learning_rate
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.episodes=episodes
        self.data_name=data_name
        self.field_size=field_size
        self.layers=layers
        self.reg_mf=reg_mf
        self.num_neg=num_neg
        self.verbose=verbose
        self.pre_train=pre_train
        self.dropout_rate=dropout_rate
        self.shuffle=shuffle
        self.use_layers=user_layers

        self.string="NFM"
        self.dataset=Dataset(data_name=self.data_name)
        self.sampler=SamplerFM(batch_size=self.batch_size,data_name=self.data_name,shuffle=self.shuffle)

        self.build_tools()
        self.feature_size=self.sampler.get_feature_size()
        self.build_net()
        self.saver = tf.train.Saver()

        self.var=[v.name for v in tf.trainable_variables()]

    def build_net(self):
        #placeholder
        self.feat_features=tf.placeholder(tf.int32,[None,None],name="user")
        self.label=tf.placeholder(tf.float32,[None,],name="label")
        self.dropout_rates=tf.placeholder(tf.float32,[None,],name="droput_rates")

        #variables
        init = tf.random_normal_initializer(0., 0.01)
        if self.pre_train:
            values = pickle.load(
                open("D:/PycharmProjects/recommend_DIY/dataset/model_values/NFM/%s.p" % self.data_name, "rb"))
            self.feature_weight=tf.Variable(values[0],name="feature_weight",dtype=tf.float32,trainable=True)
            self.feature_first=tf.Variable(values[1],name="feature_first",dtype=tf.float32,trainable=True)
        else:
            self.feature_weight=tf.get_variable(name="feature_weight",shape=[self.feature_size,self.embedding_size],dtype=tf.float32,initializer=init)
            self.feature_first=tf.get_variable(name="feature_first",shape=[self.feature_size,1],dtype=tf.float32,initializer=init)

        #embedding
        """
        [N,N_F]---embedding-->[N,N_F,E_S]--reduce_sum--->[N,E_S]--square--->[N,E_S]
                                         --square--->[N,N_F,N_S]--reduce_sum--->[N,E_S]
                                         subtract--->[N,E_S]
        """
        self.feature_embedding_part=tf.nn.embedding_lookup(self.feature_weight,self.feat_features)

        self.first_order=tf.nn.embedding_lookup(self.feature_first,self.feat_features)
        self.first_order_part=tf.reduce_sum(self.first_order,axis=1)
        self.bias=tf.Variable(tf.constant(0.0), name='bias')
        self.bias=self.bias*tf.ones_like((self.label))[:,tf.newaxis]

        #order_2
        self.sum_features_embedding=tf.reduce_sum(self.feature_embedding_part,1)
        self.sum_features_embedding_square=tf.square(self.sum_features_embedding)
        self.square_feature_embedding=tf.square(self.feature_embedding_part)
        self.square_feature_embedding_sum=tf.reduce_sum(self.square_feature_embedding,1)
        self.FM_second_order=0.5*tf.subtract(self.sum_features_embedding_square,self.square_feature_embedding_sum)

        #l2_norm
        l2_norm=tf.add_n([
            tf.reduce_sum(tf.multiply(self.feature_embedding_part,self.feature_embedding_part)),
            tf.reduce_sum(tf.multiply(self.first_order,self.first_order))
        ])

        # #order_1
        # self.FM_first_order=tf.add_n([self.first_order_part,self.bias])

        #NN
        self.FM_second_order=tf.layers.batch_normalization(self.FM_second_order,axis=1)
        self.FM_second_order=tf.nn.dropout(self.FM_second_order,self.dropout_rates[-1])
        if self.use_layers:
            for i in range(len(self.layers)):
                self.FM_second_order=tf.layers.dense(self.FM_second_order,self.layers[i],activation=None,name="layer_%d"%i,kernel_initializer=init)
                self.FM_second_order = tf.layers.batch_normalization(self.FM_second_order, axis=1)
                self.FM_second_order=tf.nn.relu(self.FM_second_order)
                self.FM_second_order=tf.nn.dropout(self.FM_second_order,self.dropout_rates[i])
        self.FM_second_order=tf.layers.dense(self.FM_second_order,1,kernel_initializer=init)
        #prediction
        self.FM=tf.reduce_sum(tf.add_n([self.first_order_part,self.bias,self.FM_second_order]),axis=1)

        #loss
        #square_loss for regression
        self.loss=tf.losses.mean_squared_error(labels=self.label,predictions=self.FM)+self.reg_mf*l2_norm
        self.square=tf.reduce_sum(tf.square(self.label-self.FM))
        #log_loss for classification
        # self.loss=tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label,self.FM))+self.reg_mf*l2_norm

        #optimizer
        # self.train_op=tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(self.loss)
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build_tools(self):
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/NFM/%s" % str(localtime))
        self.evaluate = EvaluateFM(sampler=self.sampler,logger=self.logger,model=self,data_name=self.data_name)
        self.evaluate.logging()
        self.logger.info("shuffle: %s" % self.shuffle)
        self.logger.info("layers: %s" % self.layers)
        self.logger.info("pre_train: %s" % self.pre_train)
        self.logger.info("dropout_rate: %s" % self.dropout_rate)
        self.logger.info("use_layers: %s"%self.use_layers)

    def train(self):
        for episode in range(self.episodes):
            if (1+episode)%self.verbose==0:
                test_rmse,test_time=self.evaluate.RMSE(data_name="test")
                validation_rmse,validation_time=self.evaluate.RMSE(data_name="validation")
                train_rmse,train_time=self.evaluate.RMSE(data_name="train")
                self.logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> train_rmse: %.3f  validation: %.3f     test:%.3f"%(train_rmse,validation_rmse,test_rmse))
            data_iter = self.sampler.get_train_batch()
            train_batch_number, validation_batch_number, test_batch_number = self.sampler.get_batch_number()
            total_loss=0
            trainging_start_time=timee()
            for i in range(train_batch_number):
                data=next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.feat_features:data[:,1:],
                    self.label:data[:,0],
                    self.dropout_rates:self.dropout_rate
                })
                total_loss+=loss
            self.logger.info("--Episode %d:         train_loss: %.5f     time: %.3f " % (
            episode, total_loss / train_batch_number, timee() - trainging_start_time))
            self.logger.info("")

        values=self.sess.run(self.var)
        pickle.dump((values),open("D:/PycharmProjects/recommend_DIY/dataset/model_values/NFM/%s.p" % self.data_name, "wb"))
        self.saver.save(sess=self.sess,save_path="D:/PycharmProjects/recommend_DIY/dataset/model_store/NFM/%s" % self.data_name)
        print("model has been saved!!!")

    def predict(self,user_ids,item_ids=None):
        pass

sess=tf.Session()
NFM=NFM(sess=sess,learning_rate=0.005,batch_size=4096,verbose=1,shuffle=False,data_name="ml_tag",dropout_rate=[0.8,0.7,0.5],pre_train=False,episodes=100,user_layers=True,reg_mf=0.00001,layers=[128],embedding_size=128)
sess.run(tf.global_variables_initializer())
NFM.train()