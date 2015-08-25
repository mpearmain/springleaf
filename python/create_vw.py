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

with open(projPath + 'input/xtest.vw',"wb") as outfile:
    for linenr, row in enumerate( DictReader(open(projPath + 'input/xtest_v4.csv',"rb")) ):
        # declare the contents of workspaces
        n_f = ""; n_a = ""; n_l = ""; n_j = ""; n_t = ""; n_r = ""
        label = 1
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
# vw -d ./input/xvalid.vw -i xmodel.vw -p predictions.txt

#  -l [ --learning_rate ] arg            Set learning rate
#  --power_t arg                         t power value
#   --decay_learning_rate arg             Set Decay factor for learning_rate
#                                        between passes
#  -b [ --bit_precision ] arg            number of bits in the feature table
#  --ngram arg                           Generate N grams. To generate N grams
#                                        for a single namespace 'foo', arg
#                                        should be fN.
#  -q [ --quadratic ] arg                Create and use quadratic features
#  --cubic arg                           Create and use cubic features

#  -t [ --testonly ]                     Ignore label information and just test
#  --holdout_off                         no holdout data in multiple passes
#  --holdout_period arg                  holdout period for test only, default
#                                        10
#  --holdout_after arg                   holdout after n training examples,
#                                        default off (disables holdout_period)
#  --early_terminate arg                 Specify the number of passes tolerated
#                                        when holdout loss doesn't decrease
#                                        before early termination, default is 3
#  --passes arg                          Number of Training Passes
#  --l1 arg                              l_1 lambda
#  --l2 arg                              l_2 lambda


Output options:
  -p [ --predictions ] arg              File to output predictions to
  -r [ --raw_predictions ] arg          File to output unnormalized predictions
                                        to

Reduction options, use [option] --help for more info:

  --bootstrap arg                       k-way bootstrap by online importance
                                        resampling

  --search arg                          Use learning to search,
                                        argument=maximum action id or 0 for LDF

  --cbify arg                           Convert multiclass on <k> classes into
                                        a contextual bandit problem

  --cb arg                              Use contextual bandit learning with <k>
                                        costs

  --csoaa_ldf arg                       Use one-against-all multiclass learning
                                        with label dependent features.  Specify
                                        singleline or multiline.

  --wap_ldf arg                         Use weighted all-pairs multiclass
                                        learning with label dependent features.
                                          Specify singleline or multiline.

  --csoaa arg                           One-against-all multiclass with <k>
                                        costs

  --multilabel_oaa arg                  One-against-all multilabel with <k>
                                        labels

  --log_multi arg                       Use online tree for multiclass

  --ect arg                             Error correcting tournament with <k>
                                        labels

  --oaa arg                             One-against-all multiclass with <k>
                                        labels

  --top arg                             top k recommendation

  --binary                              report loss as binary classification on
                                        -1,1

  --link arg (=identity)                Specify the link function: identity,
                                        logistic or glf1

  --stage_poly                          use stagewise polynomial feature
                                        learning

  --lrq arg                             use low rank quadratic features

  --autolink arg                        create link function with polynomial d

  --new_mf arg                          rank for reduction-based matrix
                                        factorization

  --nn arg                              Sigmoidal feedforward network with <k>
                                        hidden units

  --active                              enable active learning

  --bfgs                                use bfgs optimization

  --conjugate_gradient                  use conjugate gradient based
                                        optimization

  --lda arg                             Run lda with <int> topics

  --noop                                do no learning

  --print                               print examples

  --rank arg                            rank for matrix factorization.

  --sendto arg                          send examples to <host>

  --svrg                                Streaming Stochastic Variance Reduced
                                        Gradient

  --ftrl                                FTRL: Follow the Proximal Regularized
                                        Leader

  --pistol                              FTRL: Parameter-free Stochastic
                                        Learning

  --ksvm                                kernel svm

Gradient Descent options:
  --sgd                                 use regular stochastic gradient descent
                                        update.
  --adaptive                            use adaptive, individual learning
                                        rates.
  --invariant                           use safe/importance aware updates.
  --normalized                          use per feature normalized updates
  --sparse_l2 arg (=0)                  use per feature normalized updates

Input options:
#  -c [ --cache ]                        Use a cache.  The default is                                        <data>.cache
# --cache_file arg                      The location(s) of cache_file.#
# -k [ --kill_cache ]                   do not reuse existing cache: create a
