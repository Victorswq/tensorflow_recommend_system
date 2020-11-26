import tensorflow as tf

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='姓名')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate of training')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of training')
    parser.add_argument('--embedding_size', type=int, default=32, help='size of embedding layers')
    parser.add_argument('--verbose', type=int, default=10, help='the verbose in training')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negative item in training')
    parser.add_argument('--reg_mf', type=float, default=0.00001, help='regulation rate in training')
    parser.add_argument('--data_number', type=int, default=1, help='choose the data from 1 to 8')
    parser.add_argument('--episodes', type=int, default=1000, help='the episode of training')
    parser.add_argument('-a', '--arg', nargs='+', type=int,default=[64,32,32],help='the size of unit in layers')

    parser.add_argument('--model', type=str, default="MF", help='choose a model to train and test including MF BPR MLP NeuMF')
    args = parser.parse_args()

    if args.model=="MF":
        from general_model.MF import MF as model
    elif args.model=="BPR":
        from general_model.BPR import BPR as model
    elif args.model=="MLP":
        from general_model.MLP import MLP as model
    else:
        from general_model.NeuMF import NeuMF as model

    sess=tf.Session()
    if args.model=="BPR" or args.model=="MF":
        model_=model(sess=sess,learning_rate=args.learning_rate,reg_mf=args.reg_mf,batch_size=args.batch_size,num_neg=args.num_neg,
          verbose=args.verbose,number=args.data_number,embedding_size=args.embedding_size,episodes=args.episodes)
    else:
        model_ = model(sess=sess, learning_rate=args.learning_rate, reg_mf=args.reg_mf, batch_size=args.batch_size,episodes=args.episodes,num_neg=args.num_neg,
                       verbose=args.verbose, number=args.data_number, embedding_size=args.embedding_size,layers=args.arg)
    sess.run(tf.global_variables_initializer())
    model_.train()