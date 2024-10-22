import time
import csv
import pickle
import pandas as pd

import operator

# session_id;user_id;item_id;timeframe;eventdate

# Load .csv dataset
with open("Diginetica.csv", "rt",encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        item = data['item_id']
        curdate = data['eventdate']
        if sessid in sess_clicks.keys():
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
        if ctr % 100000 == 0:
            print ('Loaded', ctr)
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    sess_date[curid] = date

# Filter out length 1 sessions
dels=[]
for s in sess_clicks.keys():
    if len(sess_clicks[s]) == 1:
        dels.append(s)
for s in dels:
    del sess_clicks[s]
    del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks.keys():
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts.keys():
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

# filter out the items happen less than 5 times
for s in list(sess_clicks.keys()):
    curseq = sess_clicks[s]
    filseq = filter(lambda i: iid_counts[i] >= 5, curseq)
    filseq=list(filseq)
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq


# Split out test set based on dates
dates = sess_date.items()
maxdate = 1462723200.0


for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = maxdate - 86400 * 7
train_sess = filter(lambda x: x[1] < splitdate, dates)
test_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
item_ctr = 1
train_seqs = []
train_dates = []
# Convert training sessions to sequences and renumber items to start from 1
for s, date in train_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in seq:
        if i in item_dict.keys():
            outseq += [item_dict[i]]
        else:
            outseq += [item_ctr]
            item_dict[i] = item_ctr
            item_ctr += 1
    if len(outseq) < 2:  # Doesn't occur
        continue
    train_seqs += [outseq]
    train_dates += [date]

test_seqs = []
test_dates = []
# Convert test sessions to sequences, ignoring items that do not appear in training set


for s, date in test_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in seq:
        if i in item_dict.keys():
            outseq += [item_dict[i]]
    if len(outseq) < 2:
        continue
    test_seqs += [outseq]
    test_dates += [date]

counts=0

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    counts = 0
    for seq, date in zip(iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            counts+=1
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", counts)
    return out_seqs, out_dates, labs



tr_seqs, tr_dates, tr_labs = process_seqs(train_seqs,train_dates)
te_seqs, te_dates, te_labs = process_seqs(test_seqs,test_dates)
train = (tr_seqs, tr_labs)
test = (te_seqs, te_labs)

f1 = open('diginetica_train.txt', 'wb')
pickle.dump(train, f1)
f1.close()
f2 = open('diginetica_test.txt', 'wb')
pickle.dump(test, f2)
f3 = open('diginetica_max_item.txt', 'wb')
pickle.dump(item_ctr+1,f3)

f2.close()

print('Done.')