from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from sklearn import preprocessing
import numpy as np



def make_data(train_path = "../input/train.csv", test_path="../input/test.csv" ):
    train = pd.read_csv(train_path).set_index("ID")
    print("Shape of train:", np.shape(train))
    test = pd.read_csv(test_path).set_index("ID")
    print("Shape of test:", np.shape(test))

    nunique = pd.Series([train[col].nunique() for col in train.columns], index=train.columns)
    constants = nunique[nunique < 2].index.tolist()
    train = train.drop(constants, axis=1)
    test = test.drop(constants, axis=1)

    # encode string
    strings = train.dtypes == 'object';
    strings = strings[strings].index.tolist();
    encoders = {}
    for col in strings:
        encoders[col] = preprocessing.LabelEncoder()
        train[col] = encoders[col].fit_transform(train[col])
        try:
            test[col] = encoders[col].transform(test[col])
        except:
            # lazy way to incorporate the feature only if can be encoded in the test set
            del test[col]
            del train[col]

    x_train = train.drop('target',1).fillna(0)
    y_train = train.target

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(test.fillna(0)))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, test.fillna(0), test.index