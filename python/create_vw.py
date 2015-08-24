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

# True/False columns
tf_columns = ["VAR0008","VAR0009","VAR0010","VAR0011","VAR0012",
     "VAR0043","VAR0196","VAR0226","VAR0229","VAR0230","VAR0232","VAR0236","VAR0239"]
an_columns = ["VAR0001","VAR0005", "VAR0044", "VAR1934", "VAR0202", "VAR0222",
                "VAR0216","VAR0283","VAR0305","VAR0325",
                "VAR0342","VAR0352","VAR0353","VAR0354","VAR0466","VAR0467"]
loc_columns = ["VAR0237", "VAR0274", "VAR0200"]
job_columns = ["VAR0404", "VAR0493"]
time_columns = ["VAR0073","VAR0075","VAR0156","VAR0157","VAR0158","VAR0159",
               "VAR0166","VAR0167","VAR0168","VAR0169","VAR0176","VAR0177",
               "VAR0178","VAR0179","VAR0204","VAR0217"]

with open(projPath + 'input/xvalid.vw',"wb") as outfile:
    for linenr, row in enumerate( DictReader(open(projPath + 'input/xvalid_v4.csv',"rb")) ):
        # declare the contents of workspaces
        n_f = ""; n_a = ""; n_l = ""; n_j = ""; n_t = ""; n_r = ""

        for kk in row.keys():
            if kk == 'ID':                
                ID = row['ID']
            else:
                if kk == 'target':            
                    label = 2 * int(row["target"]) - 1
                elif kk in tf_columns:
                    n_f += " %s_%s"%(kk,row[kk])
                elif kk in an_columns:
                    n_a += " %s_%s"%(kk,row[kk])
                elif kk in loc_columns:
                    n_l += " %s_%s"%(kk,row[kk])
                elif kk in job_columns:
                    n_j += " %s_%s"%(kk,row[kk])
                elif kk in time_columns:
                    n_t += " %s_%s"%(kk,row[kk])
                else:
                    n_r += " %s:%s"%(kk, row[kk])
        outfile.write("%s '%s |r%s |f%s |a%s |l%s |j%s |t%s \n"%(label,ID, n_r, n_f, n_a, n_l, n_j, n_t))


# export PATH=/Users/konrad/Documents/vowpal_wabbit/vowpalwabbit:$PATH