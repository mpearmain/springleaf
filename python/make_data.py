from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def make_data(train_path, test_path):
    train = pd.read_csv(train_path).set_index("ID")
    test = pd.read_csv(test_path).set_index("ID")

    x_train = train.drop('target',1)
    y_train = train.target

    # Use Label str cols.

    names = list(x_train.columns.values)
    nonnumeric_columns = filter(lambda x: 'fac' in x, names)

    # Join the features from train and test together for label encoding
    big_X = x_train[names].append(test[names])

    # XGBoost doesn't (yet) handle categorical features automatically,
    # so we need to change them to columns of integer values.
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        big_X[feature] = le.fit_transform(big_X[feature])

    # Prepare the inputs for the model
    x_train = big_X[0:x_train.shape[0]].as_matrix()
    x_test = big_X[x_train.shape[0]::].as_matrix()

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(x_test))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, x_test, test.index