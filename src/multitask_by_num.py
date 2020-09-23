# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:09:52 2020

@author: Li Xiang

running orders:
    
*run these 2files 5 times:(each with different types)

    run multitask_1.py to get review (trainset) y_labels for 5 types
    
    run multitask_2.py to get needs data (val, test) y_labels for 5 types
    
    run multitask_train to get the model
    
    run multitask_2.py (set cur_exp_param=[..]) and this file (each time set different cur_exp_param, in this way 5 times)
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:03:04 2020

@author: Li Xiang
"""
import random 
import sys
import numpy as np
from pprint import pprint
import func_eval

from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def get_sent_embed(dataset,cur_sent_embd_type,EMBEDDING_DIM,embedding_matrix):
    """
    dataset is a 3-d tensor (datasize, MAX_SENT, MAX_SENT_LEN)
    cur_sent_embd_type in ['max','ave','concat','hier']
    """
    
    #flat each review and take the non-zero
    #word_index out for computing max
    if(cur_sent_embd_type=='max'):
        data_emb=np.zeros((dataset.shape[0],EMBEDDING_DIM))
        for i in range(dataset.shape[0]):
            wordTokens = text_to_word_sequence(dataset[i])
            review_emb=np.zeros((len(wordTokens),EMBEDDING_DIM))
            for r in range(review_emb.shape[0]):
                if(wordTokens[r] in embedding_matrix):
                    review_emb[r]=embedding_matrix[wordTokens[r]]
                else:
                    pass
            data_emb[i]=np.amax(review_emb, axis=0)
        return data_emb
    
    

review_df=pd.read_csv('../data/original_dataset_with_entity.csv')
needs_df=pd.read_csv('../data/needs_with_entity.csv')

y_train_review=[asin_map_labels[asin][0] for asin in review_df['asin'].tolist()]

y_train_needs=[asin_map_labels[asin][0] for asin in needs_df['asin'].tolist()]

if(dir().count('embeddings_index')==0):
    embeddings_index = {}
    f = open('f:\Datasets\glove.6B.300d.txt',encoding='utf-8')
    
    for line in f:
        values = line.split(' ')
        word = values[0] ## The first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
        embeddings_index[word] = coefs
    f.close()
    
X_train_need_text=get_sent_embed(needs_df['reviews'],'max',300,embeddings_index)

X_train_need_entity=np.load('../data/needs_concept_array.npy',allow_pickle=True)

X_train_need_emb=np.concatenate((X_train_need_text,X_train_need_entity),axis=1)
X_train_need, X_test_need, y_train_need, y_test_need = train_test_split(X_train_need_emb, y_train_needs, test_size=0.5, random_state=42)


train_labels=y_train_review
test_labels=y_test_need

train_labels_unique=set(train_labels)

num_labels_train_dic={}
for l in train_labels_unique:
    num_labels_train_dic[l]=train_labels.count(l)
    
pprint(num_labels_train_dic)


num_labels_test_dic={}
test_labels_unique=set(test_labels)
for l in test_labels_unique:
    num_labels_test_dic[l]=test_labels.count(l)
    
pprint(num_labels_test_dic)

#------real label: test_labels--------------------
#------predicted labels: test_pred-----------------

test_label_map_ind={}
for l in num_labels_test_dic:
    nums=[ind for ind,label in list(enumerate(test_labels)) if label==l]
    test_label_map_ind[l]=nums
top_k=5


#for cur_exp_param in y_classes.keys():
test_predict_top_k=y_classes[cur_exp_param]
test_pred=np.array(test_predict_top_k)

print('class:',cur_exp_param)
for l in test_label_map_ind:
#    test
    real_labels=np.array(test_labels)[test_label_map_ind[l]]
    real_labels=np.array([np.array([l]) for l in real_labels])
    pred_labels=test_pred[test_label_map_ind[l]]
#    pred_labels=test_pred
    print('\nlabel:',l)
    
    

    print("precision:")
    i = 0
    precisions = []
    
    while i < top_k:
        
        y_pred = pred_labels[:, 0:i+1]
        i = i+1
    
        precision = func_eval._precision_score(y_pred,real_labels)
        precisions.append(precision)
    
        print(precision)
    
    print("recall:")
    i = 0
    recalls = []
    while i < top_k:
        
        y_pred = pred_labels[:,  0:i+1]
    
        i = i+1   
        recall = func_eval.new_recall(y_pred, real_labels)
        recalls.append(recall)
    
        print(recall)
        
    print("f1:")
    f1s=[]
    for i in range(len(precisions)):
        if(precisions[i]+recalls[i]!=0):
            f1=((precisions[i]*recalls[i])/(precisions[i]+recalls[i]))
            print(f1)
        else:
            f1=0.0
            print(0.0)
        f1s.append(f1)
    
    with open('./documents/'+cur_exp_param+'_eval_by_label.txt','a') as f:
        f.write('\n\nlabel:'+str(l))
        f.write('\nprecision:')
        f.write(str(precisions))
        f.write('\nrecall:')
        f.write(str(recalls))
        f.write('\nf1:')
        f.write(str(f1s))
        
    
with open('./documents/'+cur_exp_param+'_train_labels_count.txt','w') as f:
    f.write(str(num_labels_train_dic))


with open('./documents/'+cur_exp_param+'_test_labels_count.txt','w') as f:
    f.write(str(num_labels_test_dic))

