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
import str
projPath = '/Users/konrad/Documents/projects/springleaf'
xseed = 42
todate = '20151016'


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