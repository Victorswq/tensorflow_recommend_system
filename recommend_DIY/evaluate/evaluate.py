import numpy as np
import pickle
from data.data_process import Data
from collections import defaultdict
from copy import deepcopy
from time import time

class Evaluate(object):
    def __init__(self,dataset,logger,model,data_name="ml_100k",batch_size=400):
        self.logger=logger
        self.data_name=data_name
        self.model=model
        self.dataset=dataset
        self.data=Data()
        self.batch_size=batch_size
        self.train_data=self.data.get_train_data(data_name=self.data_name)
        if self.data_name=="ml_1m" or self.data_name=="pinterest" or self.data_name=="ml_1m_lunwen":
            self.item_ids=pickle.load(open("D:/PycharmProjects/recommend_DIY/dataset/evaluate/%s.d"%self.data_name,mode="rb"))

    def get_ratings(self,test_size):
        test_data = self.data.get_test_data(data_name=self.data_name)
        user_ids = np.random.choice(test_data[:, 0],size=test_size)
        ratings = self.model.predict(user_ids)
        return user_ids,ratings

    def F1(self,ratingss,user_ids,top_k=[10,20]):
        user_movie = self.dataset.get_user_movie()
        length=len(user_ids)
        for k in top_k:
            ratings=deepcopy(ratingss)
            pres, recalls, f1s = 0, 0, 0
            idx = 0
            for user_id in user_ids:
                rating=ratings[idx]
                TP, FP, FN = 0, 0, 0
                length_ = len(user_movie[user_id])
                rank=np.argsort(rating)[-k:]
                for index in range(len(rank)):
                    if rank[index] in user_movie[user_id]:
                        TP += 1
                    else:
                        FP += 1
                FN = length_ - TP
                pre = TP / k

                recall = TP / (TP + FN)
                pres += pre
                recalls += recall
                idx+=1
            if pres==0 and recalls==0:
                f1=0
            else:
                f1=2*pres * recalls / (pres + recalls) / length
            self.logger.info(
                "-------------------------------------------------------topK %d:   pre: %.3f      recall: %.3f        F1: %.3f" % (
                k, pres / length, recalls / length, f1))

    def HitRate(self,ratingss,user_ids,top_k=[10,20]):
        user_movie = self.dataset.get_user_movie()
        length = len(user_ids)
        for k in top_k:
            ratings=deepcopy(ratingss)
            idx=0
            hr=0
            for user_id in user_ids:
                rating=ratings[idx]
                rank=np.argsort(rating)[-k:]
                for index in range(len(rank)):
                    if rank[index] in user_movie[user_id]:
                        hr+=1
                        break
                idx+=1
            self.logger.info("-------------------------------------------------------HR@%d:  %.3f" % (k,hr/length))

    def NDCG_plus(self,user_ids,ratingss,top_k=[10,20]):
        user_movie = self.dataset.get_user_movie()
        length = len(user_ids)
        for k in top_k:
            raitngs=deepcopy(ratingss)
            idx=0
            ndcg=0
            for user_id in user_ids:
                rating=raitngs[idx]
                rank=np.argsort(rating)[-k:]
                idcg=[]
                idcg.append(0)
                for idx,item in enumerate(rank):
                    if item in user_movie[user_id]:
                        idcg.append(1/np.log(k-idx+1))
                if np.max(idcg)==0:
                    continue
                idcg/=np.max(idcg)
                ndcg+=np.sum(idcg)
                idx+=1
            self.logger.info("-------------------------------------------------------NDCG@%d:  %.3f" % (k, ndcg/length))

    def MRR(self,user_ids,ratingss,top_k=[10,20]):
        user_movie=self.dataset.get_user_movie()
        length=len(user_ids)
        for k in top_k:
            ratings=deepcopy(ratingss)
            idx=0
            mrr=0
            for user_id in user_ids:
                rating=ratings[idx]
                idx+=1
                rank=np.argsort(rating)[-k:]
                mrr_=0
                f=0
                for index in range(k):
                    if rank[index] in user_movie[user_id]:
                        mrr_+=1/(k-index)
                        f+=1
                if f==0:continue
                mrr_/=f
                mrr+=mrr_
            self.logger.info(
                "-------------------------------------------------------MRR@%d:  %.3f" % (k, mrr / length))

    def RMSE(self,batch_size=400):
        test_data = self.data.get_test_data(data_name=self.data_name)
        len_test_data=len(test_data)
        batch_index=np.random.choice(len_test_data,size=batch_size)
        batch_data=test_data[batch_index,:]
        rmse=self.model.sess.run(self.model.loss,feed_dict={
            self.model.user:batch_data[:,0],
            self.model.item:batch_data[:,1],
            self.model.label:np.ones_like(batch_data[:,2])
        })
        rmse=np.sqrt(rmse)
        self.logger.info("-------------------------------------------------------RMSE: %.3f     batch_size: %d"%(rmse,batch_size))

    def HR(self,batch_size=400,top_k=[10,20]):
        test_data = self.data.get_test_data(data_name=self.data_name)
        user_ids = np.random.choice(test_data[:, 0], size=batch_size)
        item_ids=self.item_ids[user_ids]
        ratings = self.model.predict(user_ids, item_ids)
        for i in range(len(top_k)):
            hr = 0
            ratingss=deepcopy(ratings)
            for idx in range(batch_size):
                rating=ratingss[idx]
                rating[np.argsort(rating)[:-top_k[i]]] = 0
                if rating[-1]!=0:
                    hr+=1
            self.logger.info("-------------------------------------------------------HR@%d: %.3f     batch_size: %d" % (top_k[i],hr/batch_size, batch_size))

    def NDCG(self,batch_size=400,top_k=[10,20]):
        test_data = self.data.get_test_data(data_name=self.data_name)
        user_ids = np.random.choice(test_data[:, 0], size=batch_size)
        item_ids = self.item_ids[user_ids]
        ratingss = self.model.predict(user_ids, item_ids)
        for i in range(len(top_k)):
            ndcg = 0
            idcg=0
            ratings=deepcopy(ratingss)
            for idx in range(batch_size):
                idcg += 1 / np.log(2)
                rating = ratings[idx]
                rating[np.argsort(rating)[:-top_k[i]]] = 0
                if rating[-1]==0:continue
                rank=deepcopy(rating)
                rank=np.sort(rank)
                for index in reversed(range(100)):
                    if rank[index]==rating[-1]:
                        ndcg+=1/np.log(100-index+1)
                        break
                        # if index>100-top_k[i]:
                        #     ndcg+= np.log(2) / np.log(100-index+1)
                        # else:
                        #     ndcg += np.log(2) / np.log(top_k[i])
            if idcg==0:
                self.logger.info("division by zero")
            else:
                self.logger.info("-------------------------------------------------------NDCG@%d: %.3f     batch_size: %d" % (top_k[i], ndcg/idcg, batch_size))

class EvaluateFM(object):
    def __init__(self,sampler,logger,model,data_name="frappe",batch_size=400):
        self.sampler=sampler
        self.logger=logger
        self.model=model
        self.data_name=data_name
        self.batch_size=batch_size
        self.data=Data()

    def logging(self):
        self.logger.info("------------------"+str(self.model.string)+"--------------------")
        self.logger.info("learning_rate:"+str(self.model.lr))
        self.logger.info("reg_mf:"+str(self.model.reg_mf))
        self.logger.info("batch_size:"+str(self.model.batch_size))
        self.logger.info("embedding_size:"+str(self.model.embedding_size))
        self.logger.info("data_name: "+str(self.model.data_name))
        self.logger.info("number_of_epochs:"+str(self.model.episodes))
        self.logger.info("verbose:"+str(self.model.verbose))
        self.logger.info("num_neg:" + str(self.model.num_neg))

    def RMSE(self,data_name="test"):
        if data_name=="test":
            data_iter = self.sampler.get_test_batch()
            _,_, test_batch_number = self.sampler.get_batch_number()
            length=self.sampler.get_test_len()
        elif data_name=="validation":
            data_iter = self.sampler.get_validation_batch()
            _, test_batch_number,_ = self.sampler.get_batch_number()
            length=self.sampler.get_validation_len()
        else:
            data_iter = self.sampler.get_train_batch()
            test_batch_number, _,_ = self.sampler.get_batch_number()
            length=self.sampler.get_train_len()
        total_square=0
        training_start_time=time()
        for i in range(test_batch_number):
            data=next(data_iter)
            square=self.model.sess.run(self.model.square,feed_dict={
                self.model.feat_features:data[:,1:],
                self.model.label:data[:,0],
                self.model.dropout_rates:np.ones_like(self.model.layers)
            })
            total_square+=square
        total_square/=length
        total_loss=np.sqrt(total_square)
        return total_loss,time()-training_start_time

class SequentialEvaluate(object):
    def __init__(self,logger,model,data_name="frappe"):
        self.logger=logger
        self.model=model
        self.data_name=data_name


    def MRR(self,ratingss,pos_items,top_k=[10,20]):

        for k in top_k:
            ratings = deepcopy(ratingss)
            mrr = 0
            length=len(ratingss)
            for idx in range(length):
                rating = ratings[idx]
                rank = np.argsort(rating)[-k:]
                mrr_ = 0
                f = 0
                for index in range(k):
                    if rank[index] == pos_items[idx]:
                        mrr_ += 1 / (k - index)
                        f += 1
                if f == 0: continue
                mrr_ /= f
                mrr += mrr_
            self.logger.info(
                "-------------------------------------------------------MRR@%d:  %.3f" % (k, mrr / length))

    def HR(self,ratingss,pos_items,top_k=[10,20]):
        length = len(ratingss)
        for k in top_k:
            ratings = deepcopy(ratingss)
            hr = 0
            for idx in range(length):
                rating = ratings[idx]
                rank = np.argsort(rating)[-k:]
                for index in range(len(rank)):
                    if rank[index] == pos_items[idx]:
                        hr += 1
                        break
            self.logger.info("-------------------------------------------------------HR@%d:  %.3f" % (k, hr / length))

    def F1(self,ratingss,pos_items,top_k=[10,20]):
        length=len(ratingss)
        for k in top_k:
            ratings=deepcopy(ratingss)
            pres, recalls, f1s = 0, 0, 0
            for idx in range(length):
                rating=ratings[idx]
                TP, FP, FN = 0, 0, 0
                length_ = 1
                rank=np.argsort(rating)[-k:]
                for index in range(len(rank)):
                    if rank[index] == pos_items[idx]:
                        TP += 1
                    else:
                        FP += 1
                FN = length_ - TP
                pre = TP / k

                recall = TP / (TP + FN)
                pres += pre
                recalls += recall
            if pres==0 and recalls==0:
                f1=0
            else:
                f1=2*pres * recalls / (pres + recalls) / length
            self.logger.info(
                "-------------------------------------------------------topK %d:   pre: %.3f      recall: %.3f        F1: %.3f" % (
                k, pres / length, recalls / length, f1))