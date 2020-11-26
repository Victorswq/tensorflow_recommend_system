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
                 embedding_size=64,
                 verbose=1,):
        super(STAMP, self).__init__(data_name=data_name,string=string)
        self.sess=tf.Session()
        self.data_name=data_name
        self.string=string
        self.max_len=max_len
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size
        self.verbose=verbose

        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    def build_placeholder(self):
        self.S=tf.placeholder(tf.int32,[None,self.max_len],name="Session")
        self.label=tf.placeholder(tf.int32,[None,],name="label")
        self.neg_item=tf.placeholder(tf.int32,[None,],name="item_neg")
        self.attention_mask=tf.placeholder(tf.float32,[None,self.max_len],name="attention_mask")

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
        self.ba=tf.ones_like(tf.cast(self.S,tf.float32))


    def build_inference(self,item_input):
        pass


    def build_graph(self):
        batch_size = tf.shape(self.S)[0]
        self.xt=tf.nn.embedding_lookup(self.X,self.S)
        # N * max_len * embedding_size

        """
                get the w2mt
        """
        # get the mt
        self.mt=self.xt[:,-1,:]
        self.mts=tf.reshape(tf.tile(self.mt,[1,self.max_len]),[-1,self.max_len,self.embedding_size])
        # N * embedding_size ==>> N * max_len_mul_embedding_size ==>> N * max_len * embedding_size

        # get_w2
        self.w22=tf.reshape(tf.tile(self.w2,[batch_size,1]),[-1,self.embedding_size,self.embedding_size])
        # N_mul_batch_size * embedding_size ==>> N * embedding_size * embedding_size

        self.w2mt=tf.matmul(self.mts,self.w22)
        # N * max_len * embedding_size


        """
                get the w3ms
        """
        # get the ms
        self.ms=tf.reduce_mean(self.xt,axis=1)
        self.mss=tf.reshape(tf.tile(self.ms,[1,self.max_len]),[-1,self.max_len,self.embedding_size])
        # N * embedding_size ==>> N * max_len_mul_embedding_size ==>> N * max_len * embedding_size

        # get_w3
        self.w33=tf.reshape(tf.tile(self.w3,[batch_size,1]),[-1,self.embedding_size,self.embedding_size])
        # N_mul_embedding_size * embedding_size ==>> N * embedding_size * embedding_size

        self.w3ms=tf.matmul(self.mss,self.w33)
        # N * max_len * embedding_size
        """
                get the w3ms
        """

        """
                get the w1xi
        """
        self.w11=tf.reshape(tf.tile(self.w1,[batch_size,1]),[-1,self.embedding_size,self.embedding_size])
        # N_mul_embedding_size * embedding_size ==>> N * embedding_size * embedding_size
        self.w1xi=tf.matmul(self.xt,self.w11)
        # N * max_len * embedding_size


        # calculate ai
        self.ai=tf.nn.sigmoid((self.w1xi+self.w2mt+self.w3ms) * tf.expand_dims(self.attention_mask,axis=-1))
        # self.ai = tf.nn.sigmoid((self.w1xi + self.w2mt + self.w3ms) * tf.expand_dims(self.attention_mask, -1))
        # N * max_len * embedding_size

        self.w00=tf.reshape(tf.tile(self.w0,[batch_size,1]),[-1,self.embedding_size,1])
        # N_mul_embedding_size * 1 ==>> N * embedding_size * 1
        self.ai=tf.matmul(self.ai,self.w00)
        # N * max_len * 1
        # self.ai=tf.reshape(self.ai,[-1,self.max_len])
        # self.ai=tf.nn.softmax(tf.multiply(self.ai,self.attention_mask),axis=-1)
        # self.ai=tf.reshape(self.ai,[-1,self.max_len,1])

        self.ma=tf.reduce_sum(self.ai*self.xt,axis=1)
        # N * 1 * max_len ==>> N *  embedding_size

        # mlp_ms & mlp_mt
        self.hs=self.mlp_ms(self.ma) # N * embedding_size
        self.ht=self.mlp_mt(self.mt)# N * embedding_size

        # the trilinear product
        """
        situation one: only for item in the label
        """
        prod=self.hs * self.ht
        # self.pos_=tf.nn.embedding_lookup(self.X,self.label) # N * embedding_size
        # self.pos_y=tf.sigmoid(tf.reduce_sum(tf.multiply(self.pos_,prod),axis=1)) # N * 1
        # self.pos_loss=-tf.reduce_mean(tf.log(1e-24 + self.pos_y))
        #
        # self.neg_ = tf.nn.embedding_lookup(self.X, self.neg_item)  # N * embedding_size
        # self.neg_y = tf.sigmoid(tf.reduce_sum(tf.multiply(self.neg_, prod), axis=1))  # N * 1
        # self.neg_loss= -tf.reduce_mean(tf.log(1e-24 + 1 -self.neg_y))
        #
        # self.loss=self.neg_loss+self.pos_loss
        """
        situation two: for all item in the item embedding
        """
        self.y=tf.matmul(prod,self.X[1:],transpose_b=True) # N * num_items
        self.y=tf.squeeze(self.y) # N * num_items
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.y))

        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()


    def train(self):
        for episode in range(self.episodes):
            training_start_time=self.get_time()
            item_seq_list, item_pos_list,attention_mask=self.generate_sequences()
            data=DataIterator(item_seq_list,item_pos_list,attention_mask,batch_size=self.batch_size,shuffle=False)
            total_loss=0
            batch_number=0
            for item_seq,item_label,mask in data:
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.S:item_seq,
                    self.label:item_label,
                    # self.neg_item:neg_item,
                    self.attention_mask:mask
                })
                total_loss+=loss
                batch_number+=1
                if batch_number % 50 == 0:
                    print("batch %d:   loss: %.3f" % (batch_number, loss))
            self.record_train(episode, total_loss, batch_number, training_start_time)
            if episode % self.verbose == 0:
                ratings, pos_items = self.predict(batch_size=10*self.batch_size)
                self.evaluate.MRR(ratings, pos_items)
                self.evaluate.HR(ratings,pos_items)

    def predict(self, batch_size):
        item_seq_list, item_pos_list, attention_mask = self.generate_sequences(train_or_test="test")
        data = DataIterator(item_seq_list, item_pos_list, attention_mask, batch_size=batch_size, shuffle=True)
        for item_seq, item_pos, mask in data:
            ratings = self.sess.run(self.y, feed_dict={
                self.S: item_seq,
                self.attention_mask: mask,
            })
            return ratings, item_pos

stamp=STAMP(data_name="diginetica",batch_size=512,learning_rate=0.005,embedding_size=32,verbose=1,max_len=20)
stamp.train()