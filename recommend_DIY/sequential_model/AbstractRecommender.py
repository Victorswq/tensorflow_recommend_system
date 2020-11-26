
class AbstractRecommender():

    def __init__(self,learning_rate=0.001,episodes=100,verbose=5,embedding_size=64,reg_embedding=0.001,reg_layers=0.00001,batch_size=512,data_name="ml_100k",negative_num=1):
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.verbose=verbose
        self.embedding_size=embedding_size
        self.reg_embedding=reg_embedding
        self.reg_layers=reg_layers
        self.batch_size=batch_size
        self.data_name=data_name
        self.negative_num=negative_num

    def build_net(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self,user_ids,item_ids):
        raise NotImplementedError

    def build_tools(self):
        raise NotImplementedError