import numpy as np
import pandas as pd


def accuracy(ypred,ytrue): 
    if ytrue.size != ypred.size:
        raise ValueError('Incompatible size!')
    
    elif ytrue.size == 0:
        return 0
    
    else:
        return np.sqrt(np.nanmean(((ytrue - ypred) / ytrue) ** 2))


def get_acc(ypred,ytrue):
    acc_day=[]
    for jth in range(len(ytrue)//96):
        acc_day.append(accuracy(ypred[jth*96:(jth+1)*96],ytrue[jth*96:(jth+1)*96]))
    return np.nanmean(acc_day)


def get_mape(ypred, ytrue):
    acc_day=[]
    for jth in range(len(ytrue)//96):
        acc_day.append(one_mape(ypred[jth*96:(jth+1)*96], ytrue[jth*96:(jth+1)*96]))
    return np.nanmean(acc_day)


def one_mape(ypred, ytrue):
    return np.nanmean(abs(ytrue - ypred) / ytrue)