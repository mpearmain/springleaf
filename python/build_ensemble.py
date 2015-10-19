# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:44:16 2015

@author: konrad
"""

import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score
from os import listdir
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy.stats import rankdata

# import str
projPath = '/Users/konrad/Documents/projects/springleaf'
xseed = 42
todate = '20151018'


## data preparation
# list the validation set forecasts
files_list = [ projPath + '/submissions/' + f for f in listdir(projPath + '/submissions')  if f.startswith('predValid')]
# initialize an empty matrix
xvalid = np.zeros([pd.read_csv(files_list[1]).shape[0], len(files_list)])
# populate the columns
for ii in range(0,len(files_list)):
    x = pd.read_csv(files_list[ii])
    xvalid[:,ii] = np.array(x)[:,1]
# list the test set forecasts
files_list2 = [ projPath + '/submissions/' + f for f in listdir(projPath + '/submissions')  if f.startswith('predFull')]
# initialize an empty matrix
xfull = np.zeros([pd.read_csv(files_list2[1]).shape[0], len(files_list2)])
# populate the columns
for ii in range(0,len(files_list)):
    x = pd.read_csv(files_list2[ii])
    xfull[:,ii] = np.array(x)[:,1]

# grab ids and yvalid
xtrain = pd.read_csv(projPath + '/input/xtrain_v3_r3.csv')
id_train = np.array(xtrain.ID); y = np.array(xtrain.target)
xtrain.drop('ID', axis = 1, inplace = True)
xtrain.drop('target', axis = 1, inplace = True)
xtest = pd.read_csv(projPath + '/input/xtest_v3_r3.csv')
id_test = np.array(xtest.ID)
xtest.drop('ID', axis = 1, inplace = True)
del xtest

# validation subset: id and target values
xfold = pd.read_csv(projPath + '/input/xfolds.csv')
validrange = np.where(xfold.valid == 1)
trainrange = np.where(xfold.valid == 0)
yvalid = y[validrange]; id_valid = id_train[validrange]
    
## build ensemble     
nTimes = 40
idFix = StratifiedShuffleSplit(yvalid, n_iter = nTimes, test_size = 0.25, random_state = 1978)
resmat = np.zeros((nTimes, 5))
ii = 0
for train_index, test_index in idFix:
    xvalid0 = xvalid[train_index,:]
    xvalid1 = xvalid[test_index,:]
    y0 = yvalid[train_index]
    y1 = yvalid[test_index]
    x0 = np.zeros(xvalid0.shape)
    x1 = np.zeros(xvalid1.shape)
    # convert x0, x1 to ranks
    for ff in range(0, xvalid.shape[1]):
        x0[:,ff] = rankdata(xvalid0[:,ff])
        x1[:,ff] = rankdata(xvalid1[:,ff])
                
    # fit ada
    clf = AdaBoostClassifier(base_estimator=None, n_estimators=125, 
                         learning_rate=0.025, algorithm='SAMME.R', 
                         random_state=190, )
    clf.fit(x0, y0)
    pr1 = clf.predict_proba(x1)[:,1]
    resmat[ii,0] = roc_auc_score(y1, pr1)
    # bagging + ada
    clf0 = AdaBoostClassifier(base_estimator=None, n_estimators=125, 
                         learning_rate=0.025, algorithm='SAMME.R', 
                         random_state=190, )
    clf1 = BaggingClassifier(base_estimator=clf0, n_estimators=25, 
                            max_samples=0.5, max_features=0.95, bootstrap=False, 
                            bootstrap_features=False, oob_score=False, 
                            n_jobs=-1, random_state=xseed + 1, verbose=1)
    clf1.fit(x0, y0)                        
    pr2 = clf1.predict_proba(x1)[:,1]
    resmat[ii,1] = roc_auc_score(y1, pr2)
    # bagging 
    clf0 = ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, verbose = 1, 
                           class_weight = 'auto', min_samples_leaf = 5, 
                           random_state = xseed)
    clf1 = AdaBoostClassifier(base_estimator=clf0, n_estimators=25, 
                         learning_rate=0.025, algorithm='SAMME.R', 
                         random_state=190, )
    clf2 = BaggingClassifier(base_estimator=clf1, n_estimators=25, 
                            max_samples=0.25, max_features=0.9, bootstrap=False, 
                            bootstrap_features=False, oob_score=False, 
                            n_jobs=-1, random_state=xseed + 1, verbose=1)
    clf2.fit(x0, y0)                        
    pr3 = clf2.predict_proba(x1)[:,1]
    resmat[ii,2] = roc_auc_score(y1, pr3)
    
    ii = ii + 1
    