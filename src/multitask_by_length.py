#"""
#Created on Wed Jul 17 14:55:01 2019
#
#@author: Li Xiang
#
#*run these 2files 4 times:(each with different types)
#
#    run multitask_1 to get review (trainset) y_labels for 4 types
#    
#    run multitask_2 to get needs data (val, test) y_labels for 4 types
#
#variables for model training:
#    X_train, X_test
#    y_train_dic, y_test_dic 
#    
#variables for model val/testing:
#    Xgen_train, Xgen_test
#    ygen_val_dic, ygen_test_dic
#
#variables for length experiment:
#    asin_review,sent_length_lst,rank_indices
#    
#"""
#
#from keras.layers import Input, Dense
#from keras.models import Model
#import numpy as np
#
##---------set params----------------------------
#exp_param='long'#''#'short'short
#long_short_split=0.2 # long top 20% longest
#
##--------param setting ends----------------------
#
#long_rev_inds=rank_indices[:int(len(rank_indices)*long_short_split)]
#short_rev_inds=rank_indices[int(len(rank_indices)*long_short_split):]
#
##-----------params-----------------------
#top_k=5
##-------------------keras multitask model--------------------------------------
#label_count={
#    'gpu':8,
#    'ram':6,
#    'hd':11,
#    'screen':9,
#    'cpu':10
#}
#
#if(cur_sent_embd_type=='concat'):
#    embed_dimen=900#2048#1024
#else:
#    embed_dimen=1324#2048#1024
#    
#inputs = Input((embed_dimen,))
#x = Dense(100, activation='relu')(inputs)
##x = Dense(100, activation='relu')(x)
#x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()] 
#model = Model(inputs, x) 
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#
#Y_train=[y_train_dic['gpu'],y_train_dic['ram'],y_train_dic['hd'],y_train_dic['screen'],y_train_dic['cpu']]
#Y_test=[y_test_dic['gpu'],y_test_dic['ram'],y_test_dic['hd'],y_test_dic['screen'],y_test_dic['cpu']]
#Ygen_val=[ygen_val_dic['gpu'],ygen_val_dic['ram'],ygen_val_dic['hd'],ygen_val_dic['screen'],ygen_val_dic['cpu']]
#Ygen_test=[ygen_test_dic['gpu'],ygen_test_dic['ram'],ygen_test_dic['hd'],ygen_test_dic['screen'],ygen_test_dic['cpu']]
#
##-----------------------set long/short test datasets------------------------
#
#X_test_long=np.array([Xgen_test[i] for i in long_rev_inds])
#X_test_short=np.array([Xgen_test[i] for i in short_rev_inds])
#
#if(exp_param=='short'):
#    Xgen_test_new=X_test_short
#elif(exp_param=='long'):
#    Xgen_test_new=X_test_long
#
#y_test_all_labels_long={}
#y_test_all_labels_short={}
#
#for key in ygen_test_all_label_dic:
#    y_test_all_labels_long[key]=np.array([ygen_test_all_label_dic[key][i] for i in long_rev_inds])
#    y_test_all_labels_short[key]=np.array([ygen_test_all_label_dic[key][i] for i in short_rev_inds])
#
#
##------------------------------setting end-----------------------------------
#
##model.fit(x=X_train,y=Y_train,validation_data=(Xgen_train,Ygen_val),epochs=15)
#model.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),epochs=20)
#print('\n')
#model.fit(Xgen_train,Ygen_val,validation_split=0.2,epochs=20)
#
#y_pred_prob=model.predict(Xgen_test_new)
#y_classes={}
#for prediction,types in zip(y_pred_prob,list(label_count.keys())):
#    y_classes[types] = prediction.argsort()[:,::-1][:,:top_k]
#
##---------------------------evaluation------------------------------
#ndcgs={}
#precisions={}
#recalls={}
#
#for key in y_classes:
#    if(exp_param=='short'):
#        labels_to_eval=y_test_all_labels_short[key][:,:1]
#    elif(exp_param=='long'):
#        labels_to_eval=y_test_all_labels_long[key][:,:1]
#        
##    labels_to_eval=ygen_test_all_label_dic[key][:,:1]
#    test_predict_top_k=y_classes[key]
#    print("\ncur_exp_param:",key,',',exp_param,'\n')
##    print("ncdg:")
#    y_pred = test_predict_top_k[:, 0:top_k]
#    ndcg_i = func_eval._NDCG_score(y_pred,labels_to_eval)
#    ndcgs[key]=ndcg_i
#    
#    y_pred = test_predict_top_k[:, 0:top_k]
#
#    precision = func_eval._precision_score(y_pred,labels_to_eval)
#    precisions[key]=precision
#    
#    
#    y_pred = test_predict_top_k[:,  0:top_k]
#
#    recall = func_eval.new_recall(y_pred, labels_to_eval)
#    recalls[key]=recall
#    
#print("recall:")
#print(recalls)
#
#print("ndcgs:")
#print(ndcgs)
#
#print("precision:")
#print(precisions)
#
#print("f1 score:")
#f1s={}
#
#for key in recalls:
#    f1s[key]=(recalls[key]*precisions[key])/(recalls[key]+precisions[key])
#    
#print(f1s)
#
#    
##-----------------new experiment with cpu relabeled------------------------
##-----long:
##recall:
##{'gpu': 0.9813084112149533, 'ram': 1.0, 'hd': 0.9719626168224299, 'screen': 1.0, 'cpu': 0.8785046728971962}
##precision:
##{'gpu': 0.19626168224299065, 'ram': 0.2, 'hd': 0.19439252336448598, 'screen': 0.2, 'cpu': 0.17570093457943925}
##f1 score:
##{'gpu': 0.16355140186915887, 'ram': 0.16666666666666669, 'hd': 0.161993769470405, 'screen': 0.16666666666666669, 'cpu': 0.14641744548286603}
#
##-----short:
#
##recall:
##{'gpu': 0.9906976744186047, 'ram': 0.9930232558139535, 'hd': 0.9302325581395349, 'screen': 0.9976744186046511, 'cpu': 0.7674418604651163}
##precision:
##{'gpu': 0.19813953488372094, 'ram': 0.1986046511627907, 'hd': 0.18604651162790697, 'screen': 0.19953488372093023, 'cpu': 0.15348837209302327}
##f1 score:
##{'gpu': 0.16511627906976745, 'ram': 0.16550387596899224, 'hd': 0.15503875968992248, 'screen': 0.16627906976744186, 'cpu': 0.12790697674418605}
#
#
#
#
#
