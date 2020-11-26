import numpy as np
import tensorflow as tf

from util.data_iterator import DataIterator
from model.AbstractRecommender import Sequential_Model

class NARM(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="diginetica",
                 string="NARM",
                 hidden_size=32,
                 batch_size=8,
                 loss_function="cross_entropy",
                 learning_rate=0.0001,
                 verbose=1,
                 max_len=6,
                 embedding_size=32,
                 ):
        super(NARM,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.loss_function=loss_function
        self.learning_rate=learning_rate
        self.verbose=verbose
        self.max_len=max_len
        self.embedding_size=embedding_size

        self.build_model()


    def build_placeholder(self):
        self.seq_input=tf.placeholder(tf.int32,[None,self.max_len],name="seq_input")
        self.seq_mask=tf.placeholder(tf.float32,[None,self.max_len],name="seq_mask")
        self.label=tf.placeholder(tf.int32,[None,],name="label")
        self.dropout_rate=tf.placeholder(tf.float32,[None,],name="dropout_rate")

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        pad = tf.zeros([1, self.embedding_size], dtype=tf.float32, name="pad")
        self.item_matrix_gru=tf.get_variable(name="item_matrix_gru",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.item_matrix_gru=tf.concat([pad,self.item_matrix_gru[1:]],axis=0)

        self.item_matrix=tf.get_variable(name="item_matrix",shape=[self.num_items,self.embedding_size],initializer=init_w)
        self.matrix_B=tf.get_variable(name="B_matrix",shape=[2*self.hidden_size,self.embedding_size],initializer=init_w)
        self.GRU_layer=tf.keras.layers.GRU(units=self.hidden_size,return_sequences=True,return_state=True)

        self.A1=tf.get_variable(name="A1_for_ht",shape=[self.hidden_size,self.hidden_size],initializer=init_w)
        self.A2=tf.get_variable(name="A2_for_hj",shape=[self.hidden_size,self.hidden_size],initializer=init_w)
        self.V=tf.get_variable(name="V_for_mistake",shape=[self.hidden_size,1],initializer=init_w)


    def build_inference(self,item_input):
        pass

    def build_graph(self):
        batch_size=tf.shape(self.seq_input)[0]
        # inputs=tf.expand_dims(self.seq_input,axis=-1)
        inputs=tf.nn.embedding_lookup(self.item_matrix_gru,self.seq_input)
        # N * max_len * embedding_size
        inputs=tf.layers.dropout(inputs,rate=self.dropout_rate[0],name="drop_layer_1")
        self.whole_sequence_output, self.final_state=self.GRU_layer(inputs=inputs)
        # self.whole_sequence_output=tf.layers.dropout(self.whole_sequence_output,rate=self.dropout_rate[-1],name="dropout_layer_2")
        # whole sequence output: N * max_len * hidden_size
        # final_state: N * hidden_size

        """
        calculate the attention weight
        """
        A23d=tf.reshape(tf.tile(self.A2,[batch_size,1]),(batch_size,self.hidden_size,self.hidden_size),name="change_A2_for_calculate_attention_weight")
        # N_mul_hidden_size * embedding_size ==>> N * hidden_size * hidden_size
        A2hj=tf.matmul(self.whole_sequence_output,A23d) # N * max_len * hidden_size

        final_states=tf.reshape(tf.tile(self.final_state,[1,self.max_len]),(batch_size,self.max_len,self.hidden_size),name="change_final_state_for_state")
        # N * hidden_size ==>> N * max_len_mul_hidden_size ==>> N * max_len * hidden_size
        A13d=tf.reshape(tf.tile(self.A1,[batch_size,1]),(batch_size,self.hidden_size,self.hidden_size),name="change_A1_for_calculation_weight")
        final_states=tf.matmul(final_states,A13d)
        # N * max_len * hidden_size
        mask=tf.expand_dims(self.seq_mask,-1)
        # N * max_len * 1
        he=tf.nn.sigmoid((A2hj+final_states) * mask)
        # N * max_len * hidden_size
        VV=tf.reshape(tf.tile(self.V,[batch_size,1]),(batch_size,self.hidden_size,1))
        # N * hidden_size * 1
        attention_weight=tf.matmul(he,VV)
        # N * hidden_size * 1
        cl=tf.reduce_sum(attention_weight*self.whole_sequence_output,axis=1)
        # N * hidden_size
        cg=self.final_state
        # N * hidden_size
        clg=tf.concat([cl,cg],axis=1)
        # clg=cl+cg # N * hidden_size
        # N * 2_hidden_size
        clg=tf.layers.dropout(clg,rate=self.dropout_rate[-1],name="dropout_layer_between_GRU_and_bi_linear_similarity")
        clg_matmul_B=tf.matmul(clg,self.matrix_B)
        # N * embedding_size
        self.result=tf.matmul(clg_matmul_B,self.item_matrix_gru[1:,],transpose_b=True)
        # N * num_items
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.result))
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def train(self):
        dropout_rate=[0.2,0.5]
        for episode in range(1,1+self.episodes):
            item_seq_list, item_pos_list, attention_mask = self.generate_sequences(train_or_test="train")
            data = DataIterator(item_seq_list, item_pos_list,attention_mask, batch_size=self.batch_size,shuffle=False)
            total_loss = 0
            batch_number = 0
            training_start_time=self.get_time()
            for item_seq, item_label, mask in data:
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.seq_input: item_seq,
                    self.label: item_label,
                    self.seq_mask: mask,
                    self.dropout_rate:dropout_rate
                })
                total_loss+=loss
                batch_number+=1
                if batch_number % 100 ==0:
                    print("batch %d:   loss: %.3f"%(batch_number,total_loss/batch_number))
            self.record_train(episode,total_loss,batch_number,training_start_time)
            if episode%self.verbose==0:
                ratings,pos_items=self.predict(batch_size=10*self.batch_size)
                self.evaluate.MRR(ratings,pos_items)
                self.evaluate.F1(ratings,pos_items)
            self.record_train(episode, total_loss, batch_number, training_start_time)


    def predict(self,batch_size):
        dropout_rate=[1,1]
        item_seq_list, item_pos_list, attention_mask = self.generate_sequences(train_or_test="test")
        data = DataIterator(item_seq_list,item_pos_list, attention_mask, batch_size=batch_size, shuffle=True)
        for item_seq,item_pos,mask in data:
            ratings=self.sess.run(self.result,feed_dict={
                self.seq_input:item_seq,
                self.seq_mask:mask,
                self.dropout_rate:dropout_rate
            })
            return ratings,item_pos


sess=tf.Session()
narm=NARM(sess=sess,batch_size=512,verbose=1,learning_rate=0.001,max_len=6,embedding_size=50,hidden_size=32)
sess.run(tf.global_variables_initializer())
narm.train()