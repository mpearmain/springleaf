from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np


def make_data(train_path, test_path):
    train = pd.read_csv(train_path).set_index("ID")
    test = pd.read_csv(test_path)
    testids = pd.read_csv('../input/test.csv').set_index("ID")

    x_train = train.drop('target',1).fillna(0)
    y_train = train.target

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(test.fillna(0)))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, test.fillna(0), testids.index