import time
import tensorflow as tf

from time import time as timee
from evaluate.logger import Logger
from evaluate.evaluate import Evaluate
from data.dataset import Dataset

class AbstractRecommender():
    def __init__(self,learning_rate=0.001,
                 batch_size=512,
                 embedding_size=32,
                 episodes=100,
                 data_name="ml_100k",
                 layers=None,
                 verbose=1,
                 string="MF",
                 reg_mf=0.00001,
                 num_neg=1,):
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.episodes=episodes
        self.data_name=data_name
        self.layers=layers
        self.verbose=verbose
        self.string=string
        self.reg_mf=reg_mf
        self.num_neg=num_neg

        self.dataset=Dataset(data_name=self.data_name)
        self.num_items = self.dataset.get_max_movie_id() + 1
        self.num_users = self.dataset.get_max_user_id() + 1

    def build_graph(self):
        raise NotImplementedError

    def build_net(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self,user_ids,item_ids):
        raise NotImplementedError

    def l2_loss(self,*params):
        return tf.add_n([tf.nn.l2_loss(w) for w in params])

    def get_time(self):
        return timee()

    def build_tools(self):
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("D:/PycharmProjects/recommend_DIY/log/%s/%s" % (self.string, str(localtime)))
        self.evaluate = Evaluate(dataset=self.dataset, data_name=self.data_name, logger=self.logger, model=self)
        self.logging()

    def logging(self):
        self.logger.info("------------------"+str(self.string)+"--------------------")
        self.logger.info("learning_rate:"+str(self.learning_rate))
        self.logger.info("reg_mf:"+str(self.reg_mf))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("data_name: "+str(self.data_name))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("num_neg:" + str(self.num_neg))
        self.logger.info("layers:" + str(self.layers))