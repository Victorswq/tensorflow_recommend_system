import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model
from util.data_iterator import DataIterator


class NARM(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="diginetica",
                 string="NARM",
                 hidden_size=32,
                 ):
        super(NARM, self).__init__(data_name,string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.hidden_size=hidden_size

        self.build_model()


    def build_placeholder(self):
        self.seq_item=tf.placeholder(tf.int32,shape=[None,self.max_len],name="seq_item")
        self.label=tf.placeholder(tf.int32,shape=[None,],name="label")
        self.seq_mask=tf.placeholder(tf.float32,shape=[None,self.max_len],name="seq_mask")
        self.dropout_rate=tf.placeholder(tf.float32,shape=None,name="dropout_rate")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        pad_zero=tf.zeros(shape=[1,self.embedding_size],dtype=tf.float32,name="pad_for_zero")
        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.item_matrix=tf.concat([pad_zero,self.item_matrix[1:]],axis=0,name="item_matrix_with_pad_for_zero")
        self.GRU_layer=tf.keras.layers.GRU(units=self.hidden_size,kernel_initializer=init_w,return_state=True,return_sequences=True)
        self.matrix_B=tf.get_variable(name="matrix_B",shape=[2*self.hidden_size,self.embedding_size],initializer=init_w)
        self.A1=tf.get_variable(name="A1",shape=[self.hidden_size,self.hidden_size],initializer=init_w)
        self.A2=tf.get_variable(name="A2",shape=[self.hidden_size,self.hidden_size],initializer=init_w)
        # self.final_layer=tf.layers.Dense(units=1,activation=tf.nn.relu)
        self.V=tf.get_variable(name="V_for_mistake",shape=[self.hidden_size,1],initializer=init_w)

    def build_inference(self,item_input):
        pass

    def build_graph(self):
        batch_size=tf.shape(self.seq_item)[0]
        self.embedding_seq=tf.nn.embedding_lookup(self.item_matrix,self.seq_item)
        # N * max_len * embedding_size
        tf.layers.dropout(self.embedding_seq, rate=self.dropout_rate[0], name="dropout_layer_between_embedding_and_GRU")
        self.whole_seq,self.final_state=self.GRU_layer(self.embedding_seq)
        # whole seq: N * max_len * hidden_size       final_state: N * hidden_size
        # tf.layers.dropout(self.whole_seq,rate=self.dropout_rate[1],name="dropout_layer_between_GRU_and_attention")
        # A1*ht + A2*hj ==>> sigmod ==>> matmul matrix v
        A13d=tf.reshape(tf.tile(self.A1,[batch_size,1]),(batch_size,self.hidden_size,self.hidden_size))
        # A13d: A1 ==>> N_mul_hidden_size * hidden_size ==>> N * hidden_size * hidden_size
        ht3d=tf.reshape(tf.tile(self.final_state,[self.max_len,1]),(batch_size,self.max_len,self.hidden_size))
        A1ht=tf.matmul(ht3d,A13d)
        # N * max_len * hidden_size
        A23d=tf.reshape(tf.tile(self.A2,[batch_size,1]),(batch_size,self.hidden_size,self.hidden_size))
        # A23d: A2 ==>> N_mul_hidden_size * hidden_size ==>> N * hidden_size * hidden_size
        A2hj=tf.matmul(self.whole_seq,A23d)
        # N * max_len * hidden_size
        he=tf.nn.sigmoid((A1ht+A2hj) * tf.expand_dims(self.seq_mask,-1))
        # N * max_len * hidden_size _mul_ N * max_len * 1 ==>> N * max_len * hidden_size
        attention_weight=tf.reshape(tf.matmul(tf.reshape(he,[-1,self.hidden_size]),self.V),(batch_size,self.max_len,1))
        # N * max_len * 1
        attention_weight_whole_seq=attention_weight*self.whole_seq
        # N * max_len * hidden_size
        attention_weight_whole_seq=tf.reduce_sum(attention_weight_whole_seq,axis=1)
        # N * hidden_size
        attention_seq_with_final_state=tf.concat([attention_weight_whole_seq,self.final_state],axis=1)
        # N * 2_mul_hidden_size
        self.inference=tf.matmul(attention_seq_with_final_state,self.matrix_B)
        # N * embedding_size
        self.inference=tf.matmul(self.inference,self.item_matrix[1:],transpose_b=True)
        # N * num_items
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.inference))
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def train(self):
        for episode in range(1,1+self.episodes):
            total_loss=0.
            batch_number=0
            training_start_time=self.get_time()
            seq_item,item_label,item_seq_mask=self.generate_sequences(train_or_test="train")
            data=DataIterator(seq_item,item_label,item_seq_mask,batch_size=self.batch_size,shuffle=True,drop_last=False)
            for seq_item,item_label,item_seq_mask in data:
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict={
                    self.seq_item:seq_item,
                    self.label:item_label,
                    self.seq_mask:item_seq_mask
                })
                total_loss+=loss
                batch_number+=1
            self.record_train(episode,total_loss,batch_number,training_start_time)
            if episode % self.verbose==0:
                ratings,label=self.predict(batch_size=10*self.batch_size)
                self.evaluate.MRR(ratings,label)
                self.evaluate.F1(ratings,label)


    def predict(self,batch_size):
        dropout_rate=[1,1]
        seq_item,seq_label,seq_mask=self.generate_sequences(train_or_test="test")
        data=DataIterator(seq_item,seq_label,seq_mask)
        for seq_item,seq_label,seq_mask in data:
            ratings=self.sess.run(self.inference,feed_dict={
                self.seq_item:seq_item,
                self.seq_mask:seq_mask,
                self.dropout_rate:dropout_rate
            })
            return ratings,seq_label