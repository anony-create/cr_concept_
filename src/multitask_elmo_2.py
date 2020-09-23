# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:27:30 2020

@author: Li Xiang
"""


import numpy as np
import sys
import func_elmo
import func_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd

#--------------------------start parm setting--------------------------

cur_exp_param='ram'#['ram','hd','gpu','screen','cpu']
#cur_sent_embd_type='max'#['max','ave','concat']

#----------------------------end parm setting-------------------------

if(cur_sent_embd_type=='concat'):
    cur_word_dimen=2048
elif(cur_sent_embd_type=='max' or cur_sent_embd_type=='ave' ):
    cur_word_dimen=1024
    
#------------prepare for length experiment-----------------------------

df=pd.read_csv("../data/generated_reviews.csv")
all_asins=list(df['asin'].unique())

asin_review=dict()

for _asin in all_asins: 
    asin_review[_asin]=df[df.asin==_asin]['reviews'].tolist()
    
#-----------------------------prepare train/test set----------------------------

#test_asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

if(cur_exp_param=='cpu'):   
    test_asin_list=func_elmo.get_new_cpu_useful_asin()
else:
    test_asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

if(cur_exp_param=='cpu'):   
    asin_map_labels=func_elmo.get_new_cpu_label()
else:
    asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

asin_map_gene_reviews={}
sent_length_lst=[]#for length experiment 

for _asin in test_asin_list:
    review_embd=func_elmo.get_gener_review_embedding(emb_type=cur_sent_embd_type,asin=_asin)
    if(review_embd is not None):
        asin_map_gene_reviews[_asin]=review_embd
        sent_length_lst.extend(func_elmo.get_review_length(asin_review,_asin))
#asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)


#append reviews embeddings into sample
sample = np.zeros(shape=(1,cur_word_dimen))

y=[]
for _asin in asin_map_gene_reviews:
    sample=np.concatenate((sample,asin_map_gene_reviews[_asin]),axis=0)
    for i in range(asin_map_gene_reviews[_asin].shape[0]):
        y.append(asin_map_labels[_asin])
sample=sample[1:]

#sys.exit(0)

X=sample
#change the random labels in y from 0 to N
#if(y[0]!=[]):
#    all_labels=sorted(list(set(y[0])))
if(y[0]!=[]):
    if(cur_exp_param=='cpu'):
        all_labels=sorted(list(set(np.array(y).transpose().tolist()[0])))
    else:
        all_labels=sorted(list(set(y[0])))
new_map_ylabel={}
for i in range(len(all_labels)):
    new_map_ylabel[all_labels[i]]=i
y_new=[]
for each_all_labels in y:
    y_new.append(list(map(lambda x : new_map_ylabel[x],each_all_labels)))
y_array=np.array(y_new)

#y_col_count is the total label count
y_col_count=y_array.shape[1]

Xgen_train, Xgen_test, ygen_train_all_labels, ygen_test_all_labels,\
gen_train_length_lst,gen_test_length_lst= \
    train_test_split(X, y_array,sent_length_lst, test_size=0.5, random_state=33)

# rank_indices: sort indices by length in Xgen_test
rank_indices=[ind for (ind, r_len) in sorted(list(enumerate(gen_test_length_lst)),key=lambda x:x[1],reverse=True)]

X_need_rev_entity=np.load('../data/needs_concept_array.npy',allow_pickle=True)
Xgen_train_rev_entity,Xgen_test_rev_entity=train_test_split(X_need_rev_entity, test_size=0.5, random_state=33)
Xgen_train=np.concatenate((Xgen_train,Xgen_train_rev_entity),axis=1)
Xgen_test=np.concatenate((Xgen_test,Xgen_test_rev_entity),axis=1)

#sys.exit(0)

ygen_train=np.array(ygen_train_all_labels)[:,0].tolist()
ygen_test=np.array(ygen_test_all_labels)[:,0].tolist()

ygen_val_cat = np_utils.to_categorical(ygen_train)
ygen_test_cat = np_utils.to_categorical(ygen_test)

try:
    ygen_val_dic
except NameError:
    ygen_val_dic={}
    
try:
    ygen_test_dic
except NameError:
    ygen_test_dic={}

#--record the correct label
try:
    ygen_test_all_label_dic
except NameError:
    ygen_test_all_label_dic={}

#----for experiment of label frequency
try:
    test_sort_labels
except NameError:
    test_sort_labels = {}
    
if((cur_exp_param=='ram')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

elif((cur_exp_param=='hd')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    
elif((cur_exp_param=='gpu')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    
elif((cur_exp_param=='screen')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    
    
elif((cur_exp_param=='cpu')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    


