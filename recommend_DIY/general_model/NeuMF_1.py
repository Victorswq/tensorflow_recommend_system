import tensorflow as tf
import numpy as np
from time import time as timee
import time
import pickle

from data.dataset import Dataset
from data.sampler import PointSampler
from evaluate.evaluate import Evaluate
from evaluate.logger import Logger

class NeuMF_1(object):
    def __init__(self,
                 sess,
                 embedding_size=10,
                 learning_rate=0.0001,
                 batch_size=256,
                 episodes=1,
                 data_name="ml_1m",
                 layers=[128,64,32],
                 reg_mf=0.00001,
                 verbose=1,
                 num_neg=1,
                 pre_train=False,
                 dropout_rates=1,
                 shuffle=False,
                 ):
        self.sess=sess
        self.embedding_size=embedding_size
        self.lr=learning_rate
        self.batch_size=batch_size
        self.data_name=data_name
        self.layers=layers
        self.episodes=episodes
        self.verbose=verbose
        self.reg_mf=reg_mf
        self.num_neg=num_neg
        self.pre_train=pre_train
        self.dropout_rates=dropout_rates
        self.shuffle=shuffle

        self.string="NeuMF_1"
        self.dataset=Dataset(data_name=self.data_name)
        self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=self.num_neg)

        self.max_user_id_train=self.dataset.get_max_user_id()+1
        self.max_item_id_train=self.dataset.get_max_movie_id()+1
        self.max_user_id_test=self.dataset.get_max_user_id_for_test()+1
        self.max_item_id_test=self.dataset.get_max_movie_id_for_test()+1

        self.build_net()
        self.var=[v.name for v in tf.trainable_variables()]

    def build_net(self):
        # placeholder
        self.user=tf.placeholder(tf.int32,[None,],name="user")
        self.item=tf.placeholder(tf.int32,[None,],name="item")
        self.label=tf.placeholder(tf.float32,[None,],name="label")
        self.dropout_rate=tf.placeholder(tf.float32,[None,],name="dropout_rate")

        # variables
        if self.pre_train is not True:
            self.mf_user_matrix=tf.get_variable(name="mf_user_matrix",
                                                shape=[self.max_user_id_train,self.embedding_size],
                                                dtype=tf.float32)
            self.mf_item_matrix=tf.get_variable(name="mf_item_matrix",
                                                shape=[self.max_item_id_train,self.embedding_size],
                                                dtype=tf.float32)
            self.mlp_user_matrix=tf.get_variable(name="mlp_user_matrix",
                                                 shape=[self.max_user_id_train,self.layers[0]/2],
                                                 dtype=tf.float32)
            self.mlp_item_matrix=tf.get_variable(name="mlp_item_matrix",
                                                 shape=[self.max_item_id_train,self.layers[0]/2],
                                                 dtype=tf.float32)
        else:
            values=pickle.load(open("D:/PycharmProjects/recommend_DIY/dataset/model_values/NeuMF_1/%s.p"%self.data_name,"rb"))
            self.mf_user_matrix=tf.Variable(values[0],name="mf_user_matrix",dtype=tf.float32,trainable=True)
            self.mf_item_matrix=tf.Variable(values[1],name="mf_item_matrix",dtype=tf.float32,trainable=True)
            self.mlp_user_matrix=tf.Variable(values[2],name="mlp_user_matrix",dtype=tf.float32,trainable=True)
            self.mlp_item_matrix=tf.Variable(values[3],name="mlp_item_matrix",dtype=tf.float32,trainable=True)

        self.mf_user_embedding=tf.nn.embedding_lookup(self.mf_user_matrix,self.user)
        self.mf_item_embedding=tf.nn.embedding_lookup(self.mf_item_matrix,self.item)
        self.mlp_user_embedding=tf.nn.embedding_lookup(self.mlp_user_matrix,self.user)
        self.mlp_item_embedding=tf.nn.embedding_lookup(self.mlp_item_matrix,self.item)

        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(self.mf_user_embedding, self.mf_user_embedding)),
            tf.reduce_sum(tf.multiply(self.mf_item_embedding, self.mf_item_embedding)),
            tf.reduce_sum(tf.multiply(self.mlp_user_embedding, self.mlp_user_embedding)),
            tf.reduce_sum(tf.multiply(self.mlp_item_embedding, self.mlp_item_embedding)),
        ])
        # l2_norm = tf.add_n([
        #     tf.reduce_sum(tf.multiply(self.mf_user_embedding, self.mf_user_embedding)),
        #     tf.reduce_sum(tf.multiply(self.mf_item_embedding, self.mf_item_embedding)),
        # ])

        # feature_interactions and NN
        self.mf=tf.multiply(self.mf_user_embedding,self.mf_item_embedding)
        self.mlp=tf.concat([self.mlp_user_embedding,self.mlp_item_embedding],axis=1)
        init_w=tf.random_normal_initializer(0.,.01)
        for i in range(len(self.layers)):
            self.mlp=tf.layers.dense(self.mlp,units=self.layers[i],activation=tf.nn.relu,kernel_initializer=init_w,name="layer_%d"%i)
            self.mlp=tf.layers.dropout(self.mlp,rate=self.dropout_rate[i],name="dropout_layer_%d"%i)
        self.inference=tf.concat([self.mlp,self.mf],axis=1)
        # self.inference = tf.reduce_sum(tf.concat([self.mlp, self.mf], axis=1),axis=1)
        self.inference=tf.layers.dense(self.inference,units=1,activation=None,kernel_initializer=init_w,name="NeuMF_layer")
        self.inference=tf.layers.dropout(self.inference,rate=self.dropout_rate[-1],name="NeuMF_dropout_layer")
        self.inference=tf.reduce_sum(self.inference,axis=1)

        # loss and optimizer
        #MSE
        # self.loss=tf.losses.mean_squared_error(labels=self.label,predictions=self.inference)+self.reg_mf*l2_norm
        #log_loss
        # self.loss = - tf.reduce_mean(self.label * tf.log(self.inference+1e-10) + (1 - self.label) * tf.log(1 - self.inference))+self.reg_mf*l2_norm
        self.loss=tf.losses.sigmoid_cross_entropy(self.label,self.inference)+self.reg_mf*l2_norm

        if self.pre_train:
            self.train_op=tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        else:
            self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self):
        self.saver=tf.train.Saver()
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/NeuMF_1/%s" % str(localtime))
        self.evaluate=Evaluate(dataset=self.dataset,logger=self.logger,model=self,data_name=self.data_name)
        self.evaluate.logging()
        self.logger.info("layers: %s"%self.layers)
        self.logger.info("pre_train: %s"%self.pre_train)
        self.logger.info("dropout_rates: %s" % self.dropout_rates)
        # self.logger.info("dropout_rate: %s"%self.dropout_rate)
        for episode in range(1,self.episodes):
            data_iter = self.sampler.get_train_batch(self.batch_size,self.shuffle)
            self.batch_numbers = self.sampler.get_batch_number()
            total_loss=0.
            training_start_time=timee()
            for i in range(self.batch_numbers):
                data = next(data_iter)
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.user:data[:,0],
                    self.item:data[:,1],
                    self.label:data[:,2],
                    self.dropout_rate:np.zeros_like(self.layers,dtype=np.float32)/self.dropout_rates
                })
                total_loss+=loss
                if (i+1)%(self.batch_numbers//300*100)==0:
                    print("batch_number: %d/%d    loss is: %.5f    running_time: %.3f"%(i+1,self.batch_numbers,total_loss/(i+1),timee() - training_start_time))
            self.logger.info("--Episode %d:    loss: %.5f     time: %.3f" % (episode, total_loss/self.batch_numbers, timee() - training_start_time))
            if (episode+1)%self.verbose==0:
                self.evaluate.HR(top_k=[1,10,20],batch_size=3000)
                self.evaluate.NDCG(top_k=[10,20], batch_size=3000)
            # if (episode+1)%5==0:
            #     self.lr*=0.8
        values=self.sess.run(self.var)
        pickle.dump((values),open("D:/PycharmProjects/recommend_DIY/dataset/model_values/NeuMF_1/%s.p"%self.data_name,"wb"))
        self.saver.save(sess=self.sess,save_path="D:/PycharmProjects/recommend_DIY/dataset/model_store/NeuMF_1/%s"%self.data_name)
        print("model has been saved!!!")

    def predict(self,user_ids,item_ids=None):
        ratings=[]
        if item_ids is not None:
            for i in range(len(user_ids)):
                rating=self.sess.run(self.inference,feed_dict={
                    self.user:np.full(len(item_ids[i]),user_ids[i]),
                    self.item:item_ids[i],
                    self.dropout_rate:np.ones_like(self.layers)
                })
                ratings.append(rating)
        else:
            for user in user_ids:
                rating=self.sess.run(self.predicts,feed_dict={
                    self.user:np.full(self.max_item_id_test,user),
                    self.item:np.arange(self.max_item_id_test),
                    self.dropout_rate: np.ones_like(self.layers)
                })
                ratings.append(rating)
        return ratings

sess=tf.Session()
neumf=NeuMF_1(sess=sess,pre_train=False,episodes=60,data_name="ml_1m_lunwen",learning_rate=0.01,
              shuffle=True,embedding_size=16,num_neg=1,batch_size=256,reg_mf=0.000001,dropout_rates=1.5,layers=[128,64,32,16])
sess.run(tf.global_variables_initializer())
neumf.train()