from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train_val_test_split(n_sub, kfold, fold):
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        #tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr)
        #val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    #val_id = val_index[fold]

    return train_id,test_id


def kfold_train_val_holdout_test_split(n_sub, kfold, which_fold):

    # taking 20% of data as holdout test data
    fold = which_fold
    id = list(range(n_sub))
    
    import random
    random.seed(123)
    random.shuffle(id)
    
    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    #kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)
    
    test_index = list()
    train_index = list()
    
    for tr,te in kf.split(np.array(id)):
        train_index.append(tr)
        test_index.append(te)

    train_id = train_index[fold]
    test_id = test_index[fold]    
    
    # split data into 5 fold cross validation sets
    
    kf2 = KFold(n_splits=kfold, random_state=123,shuffle = True)
    
    kf_train_index = list()
    kf_val_index = list()
    
    
    for tr_2,te_2 in kf2.split(train_id):
        kf_train_index.append(tr_2)
        kf_val_index.append(te_2)
    
    kf_train_fold_1 = train_id[kf_train_index[0]]
    kf_train_fold_2 = train_id[kf_train_index[1]]
    kf_train_fold_3 = train_id[kf_train_index[2]]
    kf_train_fold_4 = train_id[kf_train_index[3]]
    kf_train_fold_5 = train_id[kf_train_index[4]]
    
    kf_val_fold_1 = train_id[kf_val_index[0]]
    kf_val_fold_2 = train_id[kf_val_index[1]]
    kf_val_fold_3 = train_id[kf_val_index[2]]
    kf_val_fold_4 = train_id[kf_val_index[3]]
    kf_val_fold_5 = train_id[kf_val_index[4]]

    return train_id,test_id,kf_train_fold_1,kf_val_fold_1,kf_train_fold_2,kf_val_fold_2,kf_train_fold_3,kf_val_fold_3,kf_train_fold_4,kf_val_fold_4,kf_train_fold_5,kf_val_fold_5