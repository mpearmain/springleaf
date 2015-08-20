# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:31:59 2014

@author: konrad
"""

import os
import subprocess
from collections import defaultdict, Counter
from datetime import datetime, date
from csv import DictReader
import math
from glob import glob
import copy

# target files

# Data locations
projPath = "/Users/konrad/Documents/projects/springleaf/"

with open(projPath + 'input/xtrain.vw',"wb") as outfile:
    for linenr, row in enumerate( DictReader(open(projPath + 'input/xtrain_v1.csv',"rb")) ):
        n_d = ""
        for kk in row.keys():
            if kk == 'ID':                
                ID = row['ID']
            else:
                if kk == 'target':            
                    label = 2 * int(row["target"]) - 1
                else:
                    n_d += " %s_%s"%(kk,row[kk])
        outfile.write("%s '%s |n%s \n"%(label,ID, n_d))

