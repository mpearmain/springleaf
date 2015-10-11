# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:28:45 2015

@author: konrad
"""

import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score

projPath = '/Users/konrad/Documents/projects/springleaf'

which_version = "v9_r4"
xseed = 42
todate = '20151011'

## data processing
#load train
train  = pd.read_csv(projPath + '/input/xtrain_'+which_version+ '.csv')
y = train.target; train.drop('target', axis = 1, inplace = True)
id = train.ID; train.drop('ID', axis = 1, inplace = True)

# separate into validation and training
xfolds = pd.read_csv(projPath + '/input/xfolds.csv')
is_valid = np.where(xfolds.valid == 1)[0]
is_train = np.where(xfolds.valid == 0)[0]
xtrain = np.array(train)[is_train,]
xvalid = np.array(train)[is_valid,]
y_train = np.array(y)[is_train]; y_valid = np.array(y)[is_valid]
del train

# load test
xtest  = pd.read_csv(projPath + '/input/xtest_'+which_version+'.csv')
id_test = xtest.ID; xtest.drop('ID', axis = 1, inplace = True)

## model fitting ##
## extra trees classifier
clf = ExtraTreesClassifier(n_estimators = 1000, n_jobs = -1, verbose = 1, 
                           class_weight = 'auto', min_samples_leaf = 5, 
                           random_state = xseed)
clf.fit(xtrain, y_train)
# generate predictions
pred_valid = clf.predict_proba(xvalid)[:,1]
pred_full = clf.predict_proba(xtest)[:,1]
# store predictions
xpred = pd.read_csv(projPath + '/submissions/predValid_datav9r7_seed45_20151004.csv')
xpred.target = pred_valid
xpred.to_csv(projPath + '/submissions/predValid_etrees_data'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)
xpred = pd.read_csv(projPath + '/submissions/predFull_datav9r7_seed45_20151004.csv')
xpred.target = pred_full
xpred.to_csv(projPath + '/submissions/predFull_etrees_data'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)

## adaboost
clf = AdaBoostClassifier(base_estimator=None, n_estimators=400, 
                         learning_rate=0.05, algorithm='SAMME.R', 
                         random_state=xseed)
clf.fit(xtrain, y_train)
# generate predictions
pred_valid = clf.predict_proba(xvalid)[:,1]
pred_full = clf.predict_proba(xtest)[:,1]
# store predictions
xpred = pd.read_csv(projPath + '/submissions/predValid_datav9r7_seed45_20151004.csv')
xpred.target = pred_valid
xpred.to_csv(projPath + '/submissions/predValid_ada_data'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)
xpred = pd.read_csv(projPath + '/submissions/predFull_datav9r7_seed45_20151004.csv')
xpred.target = pred_full
xpred.to_csv(projPath + '/submissions/predFull_ada_data'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)


## bagging
clf0 = AdaBoostClassifier(base_estimator=None, n_estimators=200, 
                         learning_rate=0.1, algorithm='SAMME.R', 
                         random_state=xseed)
clf = BaggingClassifier(base_estimator=clf0, n_estimators=25, 
                        max_samples=0.9, max_features=0.9, bootstrap=False, 
                        bootstrap_features=False, oob_score=False, 
                        n_jobs=-1, random_state=xseed + 1, verbose=1)
clf.fit(xtrain, y_train)                        
pred_valid = clf.predict_proba(xvalid)[:,1]
pred_full = clf.predict_proba(xtest)[:,1]
# store predictions
xpred = pd.read_csv(projPath + '/submissions/predValid_datav9r7_seed45_20151004.csv')
xpred.target = pred_valid
xpred.to_csv(projPath + '/submissions/predValid_bag2_data'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)
xpred = pd.read_csv(projPath + '/submissions/predFull_datav9r7_seed45_20151004.csv')
xpred.target = pred_full
xpred.to_csv(projPath + '/submissions/predFull_bag2_data'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)


clf0 = AdaBoostClassifier(base_estimator=None, n_estimators=200, 
                         learning_rate=0.1, algorithm='SAMME.R', 
                         random_state=xseed)
clf = BaggingClassifier(base_estimator=clf0, n_estimators=25, 
                        max_samples=0.9, max_features=0.9, bootstrap=False, 
                        bootstrap_features=False, oob_score=False, 
                        n_jobs=-1, random_state=xseed + 1, verbose=1)
clf.fit(xtrain, y_train)                        
pred_valid = clf.predict_proba(xvalid)[:,1]
pred_full = clf.predict_proba(xtest)[:,1]
# store predictions
xpred = pd.read_csv(projPath + '/submissions/predValid_datav9r7_seed45_20151004.csv')
xpred.target = pred_valid
xpred.to_csv(projPath + '/submissions/predValid_bag3_'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)
xpred = pd.read_csv(projPath + '/submissions/predFull_datav9r7_seed45_20151004.csv')
xpred.target = pred_full
xpred.to_csv(projPath + '/submissions/predFull_bag3_'+which_version+'_seed'+str(xseed)+'_'+todate+'.csv', index = False)


                        
# xgboost
