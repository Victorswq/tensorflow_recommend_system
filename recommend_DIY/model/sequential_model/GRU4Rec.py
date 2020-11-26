import numpy as np
import tensorflow as tf

from model.AbstractRecommender import Sequential_Model
from util.tool import log_loss,time_matrix


class GRU4Rec(Sequential_Model):
    def __init__(self,
                 sess,
                 data_name="ml_100k",
                 string="GRU4Rec",
                 loss_way="top1",
                 hidden_act="relu",
                 final_act="relu",
                 ):
        super(GRU4Rec,self).__init__(data_name=data_name,string=string)
        self.sess=sess
        self.data_name=data_name
        self.string=string

        if loss_way=="top1":
            self.loss_way=self.top1_loss
        elif loss_way=="bpr":
            self.loss_way=self.bpr_loss
        else:
            raise ValueError("There is not loss way named %s"%loss_way)

        if hidden_act=="relu":
            self.hidden_act=tf.nn.relu
        elif hidden_act=="tanh":
            self.hidden_act=tf.nn.tanh
        else:
            raise ValueError("There is not hidden_act named %s"%hidden_act)

        if final_act=="relu":
            self.final_act=tf.nn.relu
        elif final_act=="linear":
            self.final_act=tf.identity
        elif final_act=="leaky_relu":
            self.final_act=tf.nn.leaky_relu
        else:
            raise ValueError("There is not final_act named %s"%final_act)


    def bpr_loss(self,logits):
        pos_logits=tf.matrix_diag_part(logits)
        pos_logits=tf.reshape(pos_logits,shape=[-1,1])
        loss=tf.reduce_mean(-tf.log_sigmoid(pos_logits-logits))
        return loss

    def top1_loss(self,logits):
        pos_logits=tf.matrix_diag_part(logits)
        pos_logits=tf.reshape(pos_logits,shape=[-1,1])
        loss1=tf.reduce_mean(tf.sigmoid(logits-pos_logits),axis=-1)
        loss2=tf.reduce_mean(tf.sigmoid(tf.pow(logits,2)),axis=-1)-tf.squeeze(tf.sigmoid(tf.pow(pos_logits,2))/self.batch_size)
        return tf.reduce_mean(loss1+loss2)

    def init_data(self):
        time_ok=time_matrix(self.dataset.data.get_train_data(data_name=self.data_name))
        data_uit=[[row,col,time] for (row,col),time in time_ok.items()]
        data_uit.sort(key=lambda x:(x[0],x[-1]))
        data_uit=np.array(data_uit,dtype=np.int32)
        d,idx=np.unique(data_uit[:,0],return_index=True)
        offest_idx=np.zeros(len(idx)+1,dtype=np.int32)
        offest_idx[:-1]=idx
        offest_idx[-1]=len(data_uit)
        return data_uit,offest_idx


    def build_placeholder(self):
        self.x_ph=tf.placeholder(tf.int32,[self.batch_size,],name="input")
        self.y_ph=tf.placeholder(tf.int32,[self.batch_size,],name="output")
        self.state_ph=[tf.placeholder(tf.float32,[self.batch_size,n_unit],name="lyaer_%d_state"%idx) for idx,n_unit in enumerate(self.layers)]

    def build_inference(self,item_input):
        pass

    def build_variables(self):
        init_w=tf.random_normal_initializer(0.,0.01)
        init_b=tf.constant_initializer(0.)
        self.item_matrix_for_input=tf.get_variable(name="item_matrix_for_input",shape=[self.num_items,self.layers[0]],initializer=init_w)
        self.item_matrix_for_prediction=tf.get_variable(name="item_matrix_for_prediction",shape=[self.num_items,self.layers[-1]],initializer=init_w)
        self.item_bias=tf.get_variable(name="item_bias",shape=[self.num_items],initializer=init_b)

    def build_graph(self):
        cells=[tf.nn.rnn_cell.GRUCell(size,activation=self.hidden_act) for size in self.layers]
        drop_cell=[tf.nn.rnn_cell.DropoutWrapper(cell) for cell in cells]
        stacked_cell=tf.nn.rnn_cell.MultiRNNCell(drop_cell)
        inputs=tf.nn.embedding_lookup(self.item_matrix_for_input,self.x_ph)
        outputs,state=stacked_cell(inputs,state=self.state_ph)
        self.u_emb=outputs
        self.final_state=state

        # for training
        items_embed=tf.nn.embedding_lookup(self.item_matrix_for_prediction,self.y_ph)
        items_bias=tf.gather(self.item_bias,self.y_ph)

        logits=tf.matmul(outputs,items_embed,transpose_b=True)+items_bias
        logits=self.final_act(logits)
        self.loss=self.loss_way(logits)

        # reg loss
        reg_loss=self.get_l2_loss(inputs,items_embed,items_bias)
        self.loss+=self.reg_rate*reg_loss
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def build_model(self):
        self.build_tools()
        self.build_placeholder()
        self.build_variables()
        self.build_graph()

    def train(self):
        data_uit,offset_idx=self.init_data()
        data_items=data_uit[:,1]
        for episode in range(1,1+self.episodes):
            state=[np.zeros([self.batch_size,n_unit],dtype=np.float32) for n_unit in self.layers]
            user_idx=np.random.permutation(len(offset_idx)-1)
            iters=np.arange(self.batch_size,dtype=np.int32)
            maxiter=iters.max()
            start=offset_idx[user_idx[iters]]
            end=offset_idx[user_idx[iters]+1]
            finished=False
            while not finished:
                min_len=(end-start).min()
                out_idx=data_items[start]
                for i in range(min_len-2):
                    in_idx=out_idx
                    out_idx=data_items[start+i+1]
                    out_items=out_idx
                    feed={self.x_ph:in_idx,self.y_ph:out_items}
                    for l in range(len(self.layers)):
                        feed[self.state_ph[l]]=state[l]
                    _,state=self.sess.run([self.train_op,self.final_state],feed_dict=feed)
                start=start+min_len-2
                mask=np.arange(len(iters))[(end-start)<=2]
                for idx in mask:
                    maxiter+=1
                    if maxiter>=len(offset_idx)-1:
                        finished=True
                        break
                    iters[idx]=maxiter
                    start[idx]=offset_idx[user_idx[maxiter]]
                    end[idx]=offset_idx[user_idx[maxiter]+1]
                if len(mask):
                    for i in range(len(self.layers)):
                        state[i][mask]=0
            if episode%self.verbose==0:
                ratings, pos_items = self.predict(batch_size=10 * self.batch_size)
                self.evaluate.MRR(ratings, pos_items)
                self.evaluate.F1(ratings, pos_items)
            # self.record_train(episode, total_loss, batch_number, training_start_time)


    def get_user_embedding(self):
        data_uit, offset_idx = self.init_data()
        data_items = data_uit[:, 1]
        user_embedding=np.zeros(shape=[len(offset_idx) - 1,self.layers[-1]],dtype=np.int32)
        state = [np.zeros([self.batch_size, n_unit], dtype=np.float32) for n_unit in self.layers]
        user_idx = np.arange(len(offset_idx) - 1)
        iters = np.arange(self.batch_size, dtype=np.int32)
        maxiter = iters.max()
        start = offset_idx[user_idx[iters]]
        end = offset_idx[user_idx[iters] + 1]
        batch_mask=np.ones([self.batch_size],dtype=np.int32)
        while np.sum(batch_mask)>0:
            min_len = (end - start).min()
            for i in range(min_len - 1):
                in_idx = data_items[start + i ]
                feed = {self.x_ph: in_idx}
                for l in range(len(self.layers)):
                    feed[self.state_ph[l]] = state[l]
                outputs = self.sess.run(self.u_emb, feed_dict=feed)
            start = start + min_len - 1
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                user_embedding[iters[idx]] = outputs[idx]
                if maxiter >= len(offset_idx) - 1:
                    batch_mask[idx]=0
                    start[idx] = 0
                    end[idx] = offset_idx[-1]
                else:
                    maxiter += 1
                    iters[idx] = maxiter
                    start[idx] = offset_idx[user_idx[maxiter]]
                    end[idx] = offset_idx[user_idx[maxiter] + 1]
            if len(mask):
                for i in range(len(self.layers)):
                    state[i][mask] = 0
        return user_embedding

    def predict(self,batch_size):
        data_uit, offset_idx = self.init_data()
        user_embedding=self.get_user_embedding()
        item_embedding,item_bias=self.sess.run([self.item_matrix_for_prediction,self.item_bias])
        user=np.random.choice(data_uit[:,0],size=batch_size)
        data_items=data_uit[:,1]
        pos_items=data_items[offset_idx[user]+1]-1
        user_embedding=user_embedding[user]
        all_ratings=np.matmul(user_embedding,item_embedding)+item_bias
        if self.final_act == tf.nn.relu:
            all_ratings = np.maximum(all_ratings, 0)
        elif self.final_act == tf.identity:
            all_ratings = all_ratings
        elif self.final_act == tf.nn.leaky_relu:
            all_ratings = np.maximum(all_ratings, all_ratings*0.2)
        else:
            pass
        all_ratings = np.array(all_ratings, dtype=np.float32)

        return all_ratings,pos_items

sess=tf.Session()
gru4rec=GRU4Rec(sess=sess)
sess.run(tf.global_variables_initializer())
gru4rec.train()