import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model
from util.data_iterator import DataIterator


class STAMP(Sequential_Model):
    def __init__(self,
                 data_name="ml_100k",
                 string="STAMP",
                 max_len=6,
                 batch_size=16,
                 learning_rate=0.001,
                 embedding_size=64,):
        super(STAMP, self).__init__(data_name=data_name,string=string)
        self.sess=tf.Session()
        self.data_name=data_name
        self.string=string
        self.max_len=max_len
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size

        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    def build_placeholder(self):
        self.S=tf.placeholder(tf.int32,[None,self.max_len],name="Session")
        self.label=tf.placeholder(tf.int32,[None,],name="label")
        self.neg_item=tf.placeholder(tf.int32,[None,],name="item_neg")

    def build_variables(self):
        init_w1=tf.random_normal_initializer(0.,0.05) # init for layers
        init_w2=tf.random_normal_initializer(0.,0.002) # init for embedding
        init_b=tf.constant_initializer(0.)
        self.mlp_ms=tf.layers.Dense(self.embedding_size,activation=tf.nn.tanh,kernel_initializer=init_w1,bias_initializer=init_b)
        self.mlp_mt=tf.layers.Dense(self.embedding_size,activation=tf.nn.tanh,kernel_initializer=init_w1,bias_initializer=init_b)

        pad=tf.zeros([1,self.embedding_size],dtype=tf.float32,name="pad")
        self.X=tf.get_variable(name="matrix_for_session",shape=[self.num_items,self.embedding_size],initializer=init_w2)
        self.X=tf.concat([pad,self.X[1:]],axis=0)
        self.w0=tf.get_variable(name="w0",shape=[self.embedding_size,1],initializer=init_w2)
        self.w1=tf.get_variable(name="matrix_for_xi",shape=[self.embedding_size,self.embedding_size],initializer=init_w2)
        self.w2=tf.get_variable(name="matrix_for_xt",shape=[self.embedding_size,self.embedding_size],initializer=init_w2)
        self.w3=tf.get_variable(name="matrix_for_ms",shape=[self.embedding_size,self.embedding_size],initializer=init_w2)
        self.ba=tf.get_variable(name="biases",shape=[1,self.embedding_size],initializer=init_b)


    def build_inference(self,item_input):
        pass


    def predict(self,user_ids,items=None):
        pass


    def build_graph(self):
        self.xt=tf.nn.embedding_lookup(self.X,self.S)
        # N * max_len * embedding_size
        self.mt=self.xt[:,-1,:]
        # N * embedding_size

        # get the ms
        self.ms=tf.reduce_mean(self.xt,axis=1)
        # N * embedding_size


        self.w3ms=tf.matmul(self.ms,self.w3)
        # N * embedding_size matmul embedding_size * embedding_size ===>>> N * embedding_size
        """
        get the w3ms
        """


        self.w2xt=tf.matmul(self.mt,self.w2)
        self.he=tf.add(self.w3ms,self.w2xt)+self.ba
        self.he=tf.expand_dims(self.he,axis=1)
        # N * 1 * embedding_size
        """
        get the w2xt
        """


        self.w1xi=tf.reshape(tf.matmul(tf.reshape(self.xt,[-1,self.embedding_size]),self.w1),[-1,self.max_len,self.embedding_size])
        # N * max_len * embedding_size
        # calculate ai
        self.ai=tf.nn.sigmoid(self.w1xi+self.he) # N * max_len * embedding_size


        self.ai=tf.matmul(tf.reshape(self.ai,[-1,self.embedding_size]),self.w0) # N_mul_max_len * 1
        self.ai=tf.reshape(self.ai,[-1,self.max_len,1])
        # calculate ma
        self.ma=tf.reduce_sum(tf.multiply(self.ai,self.xt),axis=1) # N * embedding_size

        # mlp_ms & mlp_mt
        self.hs=self.mlp_ms(self.ma) # N * embedding_size
        self.ht=self.mlp_mt(self.mt)# N * embedding_size

        # the trilinear product
        """
        situation one: only for item in the label
        """
        self.pos_=tf.nn.embedding_lookup(self.X,self.label) # N * embedding_size
        self.pos_=tf.multiply(self.ht,self.pos_) # N * embedding_size
        self.pos_y=tf.sigmoid(tf.reduce_sum(tf.multiply(self.pos_,self.hs),axis=1)) # N * 1
        self.pos_loss=-tf.reduce_mean(tf.log(1e-24 + self.pos_y))

        self.neg_ = tf.nn.embedding_lookup(self.X, self.neg_item)  # N * embedding_size
        self.neg_ = tf.multiply(self.ht, self.neg_)  # N * embedding_size
        self.neg_y = tf.sigmoid(tf.reduce_sum(tf.multiply(self.neg_, self.hs), axis=1))  # N * 1
        self.neg_loss= -tf.reduce_mean(tf.log(1e-24 + 1 -self.neg_y))

        self.loss=self.neg_loss+self.pos_loss
        """
        situation two: for all item in the item embedding
        """
        # self.hs = tf.expand_dims(self.hs, axis=1)  # N * 1 * embedding_size
        # self.ht = tf.expand_dims(self.ht, axis=1)  # N * 1 * embedding_size
        # self.y=tf.tile(tf.expand_dims(self.X,axis=0),[tf.shape(self.hs)[0],1,1]) # N * num_items * embedding_size
        # self.y=self.y*self.ht # N * num_items * embedding_size
        # self.y=tf.matmul(self.hs,tf.transpose(self.y,[0,2,1])) # N * 1 * num_items
        # self.y=tf.squeeze(self.y) # N * num_items
        # self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.y))
        # loss and optimizer

        self.train_op=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)


    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()


    def train(self):
        for episode in range(self.episodes):
            session_id_list, item_seq_list, item_pos_list, item_neg_list=self.generate_sequence()
            data=DataIterator(item_seq_list,item_pos_list,item_neg_list,batch_size=self.batch_size,shuffle=False)
            total_loss=0
            batch_number=0
            for item_seq,item_label,neg_item in data:
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.S:item_seq,
                    self.label:item_label,
                    self.neg_item:neg_item
                })
                total_loss+=loss
                batch_number+=1
            self.logger.info("Loss is %.3f"%(total_loss/batch_number))


stamp=STAMP(data_name="diginetica",batch_size=512,learning_rate=0.01,embedding_size=64)
stamp.train()