# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:28:45 2015

@author: konrad
"""

import pandas as pd
import numpy as np 
from sklearn import preprocessing
projPath = '/Users/konrad/Documents/projects/springleaf'

#load train and process
train  = pd.read_csv(projPath + '/input/xtrain_v8.csv')


y = train.target
