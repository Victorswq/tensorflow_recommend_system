import time
import csv
import pickle
import pandas as pd

import operator

# load .csv dataset
with open(" ","rt",encoding="utf-8") as file: # open the target csv file
    reader=csv.DictReader(file,delimiter=";")
    sess_clicks={}
    sess_date={}
    ctr=0
    curid=-1
    curdate=None
    for data in reader:
        sessid=data["session_id"]
        if curdate and not curid==sessid:
            date=time.mktime(time.strptime(curdate,"%Y-%m-%d"))
            sess_date[curid]=date
        curid=sessid
        item=date["item_id"]
        curdate=date["eventdate"]
        if sessid in sess_clicks.keys():
            sess_clicks[sessid]+=[item]
        else:
            sess_clicks[sessid]=[item]
        ctr+=1
        if ctr%100000==0:
            print("Loaded ",ctr)
    date=time.mktime(time.strptime(curdate,"%Y-%m-%d"))
    sess_date[curid]=date

# filter out length 1 sessions
dels=[]
for s in sess_clicks.keys():
    if len(sess_clicks[s])==1:
        dels.append(s)
for s in dels:
    del sess_clicks[s]
    del sess_date[s]

# count number of times each item appears
iid_counts={}
for s in sess_clicks.keys():
    seq=sess_clicks[s]
    for iid in seq:
        if iid in iid_counts.keys():
            iid_counts[iid]+=1
        else:
            iid_counts[iid]=1

sorted_counts=sorted(iid_counts.items(),key=operator.itemgetter(1))

# filter out the items happen less than 5 times
for s in list(sess_clicks.keys()):
    curseq=sess_clicks[s]
    filseq=filter(lambda i: iid_counts[i]>=5,curseq)
    filseq=list(filseq)
    if len(filseq)<2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s]=filseq


# split out test set based on dates
dates=sess_date.items()
maxdata=0

for _,data in dates:
    if maxdata<date:
        maxdata=date

# 7 days for test
splitdate=maxdata-86400*7
train_sess=filter(lambda x: x[1]<splitdate,dates)
test_sess=filter(lambda x:x[1]>splitdate,dates)

# sort sessions by date
train_sess=sorted(train_sess,key=operator.itemgetter(1))
test_sess=sorted(test_sess,key=operator.itemgetter(1))

# choosing item count >=5 gives
item_dict={}
item_ctr=1
train_sess=[]
test_sess=[]

