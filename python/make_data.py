from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
from sklearn import preprocessing



def make_data():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

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

    features = train.select_dtypes(include=['float']).columns
    features = np.setdiff1d(features, ['ID', 'target'])

    test_ids = test.ID
    y_train = train.target

    x_train = train[features].fillna(0)
    x_test = test[features].fillna(0)

    return x_train, y_train, x_test, test_ids