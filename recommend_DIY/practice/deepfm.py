import tensorflow as tf
import numpy as np
import sys
from build_data import load_data


class Args():
    feature_size=100
    field_size=15
    embedding_size=256
    deep_layers=[512,256,128]
    epoch=3
    batch_size=64
    learning_rate=0.01
    l2_reg_rate=0.01
    checkpoint_dir = 'D:/PycharmProjects/recommend_DIY/dataset/ckpt'
    is_training=True
    deep_activation=tf.nn.relu


class model():
    def __init__(self,args):
        self.feature_sizes=args.feature_size
        self.field_size=args.field_size
        self.embedding_size=args.embedding_size
        self.deep_layers=args.deep_layers
        self.l2_reg_rate=args.l2_reg_rate

        self.epoch=args.epoch
        self.batch_size=args.batch_size
        self.learning_rate=args.learning_rate
        self.deep_activation=args.deep_activation
        self.weight=dict()
        self.checkpoint_dir=args.checkpoint_dir

        self.build_model()

    def build_model(self):
        self.feat_index=tf.placeholder(tf.int32,shape=[None,None],name="feature_index")
        self.feat_value=tf.placeholder(tf.float32,shape=[None,None],name="feature_value")
        self.label=tf.placeholder(tf.float32,shape=[None,None],name="label")

        init = tf.random_normal_initializer(0., .1)
        # feature-->feature vector
        self.weight["feature_weight"]=tf.get_variable(
            name="feature_weight",
            shape=[self.feature_sizes,self.embedding_size],
            initializer=init
        )

        # order-1 in fm
        self.weight["feature_first"]=tf.get_variable(
            name="feature_first",
            shape=[self.feature_sizes,1],
            initializer=init
        )

        num_layer=len(self.deep_layers)
        # deep_layers
        input_size=self.field_size*self.embedding_size
        self.weight["layer_0"]=tf.get_variable(
            name="layer_0",
            shape=[input_size,self.deep_layers[0]],
            initializer=init
        )

        # embedding--->feat_index[N,F]--embedding-->[N,F,K]---mul--->feat_value[N,F,K]
        self.embedding_index=tf.nn.embedding_lookup(self.weight["feature_weight"],
                                                    self.feat_index)
        self.embedding_part=tf.multiply(self.embedding_index,
                                        tf.reshape(self.feat_value,[-1,self.field_size,1]))

        # order-1----->feat_index[N,F]---embedding--->[N,F,1]--mul--->feat_value[N,F,1]---reduce_sum--->[N,F]
        self.embedding_first=tf.nn.embedding_lookup(self.weight["feature_first"],
                                                    self.feat_index)
        self.embedding_first=tf.multiply(self.embedding_index,tf.reshape(self.feat_value,[-1,self.field_size,1]))
        self.first_order=tf.reduce_sum(self.embedding_first,axis=2)

        # order-2
        # embedding_part[N,F,K]---reduce_sum--->[N,K]--square---->[N,K]
        # embedding-part[N,F,K]---square--->[N,F,K]--reduce_sum-->[N,K]
        # concat second_order and first_order--------->[N,F+K]
        self.sum_second_order=tf.reduce_sum(self.embedding_part,axis=1)
        self.sum_second_order_square=tf.square(self.sum_second_order)

        self.square_second_oreder=tf.square(self.embedding_part)
        self.square_second_oreder_sum=tf.reduce_sum(self.square_second_oreder,axis=1)

        self.second_order=0.5*tf.subtract(self.sum_second_order_square,self.square_second_oreder_sum)
        self.fm_part=tf.concat([self.first_order,self.second_order],axis=1)

        # deep_part--embedding_part--reshape--->deep_embedding[N,F*K]
        # deep_embedding[N,F*K]----layer_one -->[N,layer_1]---......--->[N,layer_last]
        self.deep_embedding=tf.reshape(self.embedding_part,[-1,self.field_size*self.embedding_size])
        print(self.deep_embedding)
        for idx in range(len(self.deep_layers)):
            self.deep_embedding=tf.layers.dense(
                inputs=self.deep_embedding,
                units=self.deep_layers[idx],
                activation=tf.nn.relu,
                name="layer_size_%d"%idx,
                kernel_initializer=init,
            )

        # fm_part and deep_embedding----concat------>[N,F+K+layer_last]
        din_all=tf.concat([self.fm_part,self.deep_embedding],axis=1)
        self.out=tf.layers.dense(
            inputs=din_all,
            units=1,
            activation=tf.nn.sigmoid,
            name="concat_layer",
            kernel_initializer=init,
        )

        # loss and optimizer
        self.loss=tf.losses.log_loss(labels=self.label,predictions=self.out)
        self.global_step=tf.Variable(0,trainable=False)
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        return loss, step

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])

if __name__ == '__main__':
    args = Args()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data = load_data()
    args.feature_sizes = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True

    with tf.Session(config=gpu_config) as sess:
        Model = model(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all:%s' % cnt)
        sys.stdout.flush()
        if args.is_training:
            for i in range(args.epoch):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step = Model.train(sess, X_index, X_value, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        Model.save(sess, args.checkpoint_dir)
        else:
            Model.restore(sess, args.checkpoint_dir)
            for j in range(0, cnt):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                result = Model.predict(sess, X_index, X_value)
                print(result)