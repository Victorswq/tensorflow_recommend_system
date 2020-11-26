import os
import time
import numpy as np
import tensorflow as tf

from util.tool import pad_sequence,pad_sequences
from time import time as timee
from evaluate.logger import Logger
from evaluate.evaluate import SequentialEvaluate as Evaluate
from data.dataset import Dataset
from data.sampler import PointSampler,PairwiseSampler

class Sequential_Model(object):
    def __init__(self,
                 data_name="ml_100k",
                 string="MFF",
                 episodes=100,
                 batch_size=512,
                 learning_rate=0.001,
                 verbose=5,
                 embedding_size=32,
                 reg_rate=0.00001,
                 num_neg=1,
                 layers=None,
                 loss_function="bpr",
                 is_pairwise=False,
                 trainer="adam",
                 max_len=5,
                 ):
        self.data_name=data_name # the name of data to be used
        self.string=string # the name of the model
        self.episodes=episodes # episodes for training
        self.batch_size=batch_size # batch_size for training
        self.learning_rate=learning_rate # learning_rate
        self.verbose=verbose # for every verbose episodes evaluate the model
        self.embedding_size=embedding_size # embedding size for item or user
        self.reg_rate=reg_rate # the rate of regulation
        self.num_neg=num_neg # the number of the negative example for every positive example
        self.layers=layers # layers if use the NN
        self.loss_function=loss_function # loss function for training
        self.is_pairwise=is_pairwise # if takes the pairwise into consideration
        self.trainer=trainer # the choose of optimizer
        self.max_len=max_len # the max sequence length

        self.dataset = Dataset(data_name=self.data_name) # data for training, validation, test
        self.num_items = self.dataset.data.get_max_item(data_name=self.data_name) # the number of the items in the data
        self.num_users = 0 # the number of the user in the data

    def build_placeholder(self): # placeholder in model may be not used in tensorflow 2.0, the reason I still use the tensorflow 1.1 is the running seeped is faster than tensorflow 2.0
        raise NotImplementedError

    def build_variables(self): # variables in model
        raise NotImplementedError

    def build_inference(self,item_input): # the result of the model
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_graph(self): # the graph of the model in tensorflow
        raise NotImplementedError

    def predict(self,batch_size): # predict for test and validation
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def get_time(self):
        return timee()

    def generate_sequences(self,train_or_test="train"):
        if train_or_test=="train":
            user_pos_train=self.dataset.data.get_train_data(data_name=self.data_name)
        else:
            user_pos_train=self.dataset.data.get_test_data(data_name=self.data_name)
        train_data,labels=user_pos_train[0],user_pos_train[1]
        self.user_test_seq = {}
        item_seq_list, item_pos_list = [], []

        for data,label in zip(train_data,labels):
            item_seq_list.append(data)
            item_pos_list.append(label-1) # start from 0
        item_seq_list,item_seq_mask=pad_sequences(item_seq_list,value=0,max_len=self.max_len,padding="pre",truncating="pre")


        return item_seq_list, item_pos_list ,item_seq_mask

    def generate_sequence(self):
        self.session_pos_train=self.dataset.data.get_train_data(data_name=self.data_name)
        self.test_seq = {}
        self.test_mask={}
        user_list, item_seq_list, item_pos_list, item_neg_list, attention_mask = [], [], [], [], []
        seq_len = self.max_len
        uni_users = np.unique(list(self.session_pos_train.keys()))
        for user in uni_users:
            seq_items = self.session_pos_train[user]
            for i in range(1,len(seq_items)-1):
                if user not in self.test_seq:
                    lists = seq_items[-self.max_len-1:]
                    x,mask=pad_sequence(lists,value=0,max_len=seq_len+1,padding="pre",truncating="pre")
                    self.test_seq[user]=x
                    self.test_mask[user]=mask
                else:
                    user_list.append(user)
                    lists=seq_items[:i]
                    x,mask=pad_sequence(lists,value=0,max_len=seq_len,padding="pre",truncating="pre")
                    item_seq_list.append(x)
                    attention_mask.append(mask)
                    item_pos_list.append(seq_items[i])
                    # add the negative sample
                    negative_user=np.random.choice(uni_users,1)[0]
                    while negative_user == user:
                        negative_user = np.random.choice(uni_users, 1)[0]
                    item_neg_list.append(negative_user)

        user_list=np.stack(user_list,axis=0)
        item_seq_list=np.stack(item_seq_list,axis=0)
        item_pos_list=np.stack(item_pos_list,axis=0)
        item_neg_list=np.stack(item_neg_list,axis=0)
        attention_mask=np.stack(attention_mask,axis=0)
        self.test_seq=np.stack(self.test_seq,axis=0)
        return user_list, item_seq_list, item_pos_list,item_neg_list,attention_mask

    def record_train(self,episode,total_loss,batch_number,training_start_time):
        self.logger.info("Episode %d:     loss: %.4f     spend_time: %.3f"%(episode,total_loss/batch_number,self.get_time()-training_start_time))

    def get_l2_loss(self,*params): # l2_norm for a better training
        return tf.add_n([tf.nn.l2_loss(item) for item in params])

    def pairwise_loss(self,positive_inference,negative_inference,loss_function="bpr"):
        if loss_function=="bpr":
            # y = log(1 / (1 + exp(-x)))
            return -tf.reduce_mean(tf.log_sigmoid(positive_inference-negative_inference))
        elif loss_function=="square":
            return -tf.reduce_mean(tf.square(positive_inference-negative_inference))
        else:
            raise Exception("Please choose a right loss function")

    def pointwise_loss(self,label,prediction,loss_function="square"):
        if loss_function=="square":
            return tf.reduce_mean(tf.square(prediction-label))
        elif loss_function=="cross_entropy":
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            return tf.losses.sigmoid_cross_entropy(label,prediction)
        else:
            raise Exception("Please choose a right loss function")

    def optimizer(self,loss,trainer):
        if trainer=="adam":
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        elif trainer=="sgd":
            return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        else:
            raise Exception("Please choose a right optimizer")

    def build_tools(self):
        """
        some modules used in model
        sampler: to sampler the training data and validation data and test data
        logger: to record the running record
        evaluate: to evaluate the model
        """
        if self.is_pairwise is True:
            self.sampler=PairwiseSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=1)
        else:
            self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=self.num_neg)

        road=os.path.abspath(os.path.join(os.getcwd(), "../.."))
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("%s\log\%s\%s" % (road,self.string, str(localtime)))
        print("%s/log/%s/%s" % (road,self.string, str(localtime)))
        self.evaluate = Evaluate(data_name=self.data_name, logger=self.logger, model=self)
        self.logging()

    def logging(self):
        self.logger.info("------------------"+str(self.string)+"--------------------")
        self.logger.info("learning_rate:"+str(self.learning_rate))
        self.logger.info("reg_mf:"+str(self.reg_rate))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("num_neg:" + str(self.num_neg))
        self.logger.info("layers:" + str(self.layers))
        self.logger.info("loss function:"+str(self.loss_function))
        self.logger.info("data_name: " + str(self.data_name))
        self.logger.info("num_user:"+str(self.num_users))
        self.logger.info("num_items:"+str(self.num_items))
        self.logger.info("max_len:" + str(self.max_len))

class General_Model(object):
    def __init__(self,
                 data_name="ml_100k",
                 string="MFF",
                 episodes=100,
                 batch_size=512,
                 learning_rate=0.001,
                 verbose=5,
                 embedding_size=32,
                 reg_rate=0.00001,
                 num_neg=1,
                 layers=None,
                 loss_function="bpr",
                 is_pairwise=False,
                 trainer="adam",
                 ):
        self.data_name=data_name # the name of data to be used
        self.string=string # the name of the model
        self.episodes=episodes # episodes for training
        self.batch_size=batch_size # batch_size for training
        self.learning_rate=learning_rate # learning_rate
        self.verbose=verbose # for every verbose episodes evaluate the model
        self.embedding_size=embedding_size # embedding size for item or user
        self.reg_rate=reg_rate # the rate of regulation
        self.num_neg=num_neg # the number of the negative example for every positive example
        self.layers=layers # layers if use the NN
        self.loss_function=loss_function # loss function for training
        self.is_pairwise=is_pairwise # if takes the pairwise into consideration
        self.trainer=trainer

        self.dataset = Dataset(data_name=self.data_name) # data for training, validation, test
        self.num_items = self.dataset.get_max_movie_id() + 1 # the number of the items in the data
        self.num_users = self.dataset.get_max_user_id() + 1 # the number of the user in the data

    def build_placeholder(self): # placeholder in model may be not used in tensorflow 2.0, the reason I still use the tensorflow 1.1 is the running seeped is faster than tensorflow 2.0
        raise NotImplementedError

    def build_variables(self): # variables in model
        raise NotImplementedError

    def build_inference(self,item_input): # the result of the model
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_graph(self): # the graph of the model in tensorflow
        raise NotImplementedError

    def predict(self,user_ids,items=None): # predict for test and validation
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def get_time(self):
        return timee()

    def record_train(self,episode,total_loss,batch_number,training_start_time):
        self.logger.info("Episode %d:     loss: %.4f     spend_time: %.3f"%(episode,total_loss/batch_number,self.get_time()-training_start_time))

    def get_l2_loss(self,*params): # l2_norm for a better training
        return tf.add_n([tf.nn.l2_loss(item) for item in params])

    def pairwise_loss(self,positive_inference,negative_inference,loss_function="bpr"):
        if loss_function=="bpr":
            # y = log(1 / (1 + exp(-x)))
            return -tf.reduce_mean(tf.log_sigmoid(positive_inference-negative_inference))
        elif loss_function=="square":
            return -tf.reduce_mean(tf.square(positive_inference-negative_inference))
        else:
            raise Exception("Please choose a right loss function")

    def pointwise_loss(self,label,prediction,loss_function="square"):
        if loss_function=="square":
            return tf.reduce_mean(tf.square(prediction-label))
        elif loss_function=="cross_entropy":
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            return tf.losses.sigmoid_cross_entropy(label,prediction)
        else:
            raise Exception("Please choose a right loss function")

    def optimizer(self,loss,trainer):
        if trainer=="adam":
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        elif trainer=="sgd":
            return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        else:
            raise Exception("Please choose a right optimizer")

    def build_tools(self):
        """
        some modules used in model
        sampler: to sampler the training data and validation data and test data
        logger: to record the running record
        evaluate: to evaluate the model
        """
        if self.is_pairwise is True:
            self.sampler=PairwiseSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=1)
        else:
            self.sampler=PointSampler(batch_size=self.batch_size,data_name=self.data_name,num_neg=self.num_neg)

        road=os.path.abspath(os.path.join(os.getcwd(), "../.."))
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("%s\log\%s\%s" % (road,self.string, str(localtime)))
        print("%s/log/%s/%s" % (road,self.string, str(localtime)))
        self.evaluate = Evaluate(dataset=self.dataset, data_name=self.data_name, logger=self.logger, model=self)
        self.logging()

    def logging(self):
        self.logger.info("------------------"+str(self.string)+"--------------------")
        self.logger.info("learning_rate:"+str(self.learning_rate))
        self.logger.info("reg_mf:"+str(self.reg_rate))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("num_neg:" + str(self.num_neg))
        self.logger.info("layers:" + str(self.layers))
        self.logger.info("loss function:"+str(self.loss_function))
        self.logger.info("data_name: " + str(self.data_name))
        self.logger.info("num_user:"+str(self.num_users))
        self.logger.info("num_items:"+str(self.num_items))

# def build_tools():
#     print("yes")
#     road=os.path.abspath('..')
#     localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#     print(("%s/log/%s/%s" % (road,"MF", str(localtime))))
#     print(localtime)
#
# build_tools()
# print (os.path.abspath(os.path.join(os.getcwd(), "../..")))
