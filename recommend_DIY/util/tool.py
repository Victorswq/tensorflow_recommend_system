import tensorflow as tf
import numpy as np
from inspect import signature
from functools import wraps
import heapq
import itertools
import time
import pandas as pd
from scipy.sparse import csr_matrix
import pickle


def activation_function(act,act_input):
    act_func=None
    if act == "sigmod":
        act_func=tf.nn.sigmoid(act_input)
    elif act=="relu":
        act_func=tf.nn.relu(act_input)
    else:
        act_func=tf.nn.tanh(act_input)
    return act_func


def get_data_format(data_format):
    if data_format=="UIRT":
        columns=["user","item","rating","time"]
    elif data_format=="UIR":
        columns=["user","item","rating"]
    elif data_format=="UIT":
        columns=["user","item","time"]
    elif data_format=="UI":
        columns=["user","item"]
    else:
        raise ValueError("Please choose a correct data format!!!")
    return columns


def csr_to_user_dict(train_matrix):
    train_dict={}
    for idx,value in enumerate(train_matrix):
        if any(value.indices):
            train_dict[idx]=value.indices.copy().tolist()
    return train_dict


def csr_to_user_dict_bytime(time_matrix,train_matrix):
    train_dict={}
    time_matrix=time_matrix
    user_pos_items=csr_to_user_dict(train_matrix)
    for u,items in user_pos_items.items():
        sorted_items=sorted(items,key=lambda x:time_matrix[u,x])
        train_dict[u]=np.array(sorted_items,dtype=np.int32).tolist()
    return train_dict


def get_initializer(init_method,stddev):
    if init_method=="normal":
        return tf.random_normal_initializer(stddev=stddev)
    else:
        return tf.constant_initializer(0.)


def noise_validator(noise,allowed_noises):
    try:
        if noise in allowed_noises:
            return True
        elif noise.spilt('-')[0]=="mask" and float(noise.spilt('-')[1]):
            t=float(noise.spilt('-')[1])
            if t>=0.0 and t<=1.0:
                return True
            else:
                return False
    except:
        return False
    pass


def argmax_top_k(a,top_k=50):
    ele_idx=heapq.nlargest(top_k,zip(a,itertools.count()))
    return np.array([idx for ele,idx in ele_idx],dtype=np.int32)


def pad_sequence(sequence,value=0,max_len=None,padding="post",truncating="post",dtype=np.int32):
    pad=-20000
    x=np.full(max_len,value,dtype=dtype)
    mask=np.full(max_len,pad,dtype=np.float32)
    if max_len is None:
        max_len=len(sequence)
    if truncating=="pre":
        trunc=sequence[-max_len:]
    elif truncating=="post":
        trunc=sequence[:max_len]
    else:
        raise ValueError("Please choose a correct truncating including pre and post")

    if padding=="post":
        x[:len(trunc)]=trunc
        mask[:len(trunc)]=np.ones_like(trunc)
    elif padding=="pre":
        x[-len(trunc):]=trunc
        mask[-len(trunc):]=np.ones_like(trunc)
    else:
        raise ValueError("Please choose a correct padding including pre and post")
    return x,mask


def pad_sequences(sequences,value=0,max_len=None,padding="post",truncating="post",dtype=np.int32):
    inf_mask=-100000
    if max_len is None:
        max_len=np.max([len(x) for x in sequences])
    x=np.full([len(sequences),max_len],value,dtype=dtype)
    mask=np.full([len(sequences),max_len],inf_mask,dtype=dtype)

    for idx,s in enumerate(sequences):
        if not len(s):
            continue
        if truncating=="pre":
            trunc=s[-max_len:]
        elif truncating=="post":
            trunc=s[:max_len]
        else:
            raise ValueError("Truncating type %s not understood"%truncating)

        if padding=="post":
            x[idx,:len(trunc)]=trunc
            mask[idx,:len(trunc)]=np.ones_like(trunc)
        elif padding=="pre":
            x[idx,-len(trunc):]=trunc
            mask[idx,-len(trunc):]=np.ones_like(trunc)
        else:
            raise ValueError("Padding type %s not understood"%padding)

    return x,mask


def inner_product(a,b,name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a,b),axis=-1)


def log_loss(yij,name="log_loss"):
    with tf.name_scope(name):
        return -tf.log_sigmoid(yij)

def train_matrix(train_data):
    train_matrix = csr_matrix((np.ones_like(train_data[:, 2]), (train_data[:, 0], train_data[:, 1])), shape=(np.max(train_data[:, 0]).astype(int) + 1, np.max(train_data[:, 1]).astype(int) + 1))
    return train_matrix

def time_matrix(train_data):
    time_matrix = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])),shape=(np.max(train_data[:, 0]).astype(int) + 1, np.max(train_data[:, 1]).astype(int) + 1))
    return time_matrix