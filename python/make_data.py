from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction import DictVectorizer

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,
    and the fitted vectorizor.
    Modified from https://gist.github.com/kljensen/5452382
    """
    vec = DictVectorizer()
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(orient='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

def make_data(train_path, test_path):
    train = pd.read_csv(train_path).set_index("ID")
    test = pd.read_csv(test_path).set_index("ID")
    x_train = train.drop('target',1)
    y_train = train.target

    # Use Label str cols.
    names = list(x_train.columns.values)
    fac_columns = filter(lambda x: 'fac' in x, names)

    # Remove facVAR###dmp cols from list
    nonnumeric_columns = filter(lambda x: 'dmp' not in x, fac_columns)
    print(nonnumeric_columns)
    # Join the features from train and test together for label encoding
    big_X = x_train[names].append(test[names])

    # XGBoost doesn't (yet) handle categorical features automatically,
    # so we need to change them to columns of integer values.
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        big_X[feature] = le.fit_transform(big_X[feature])

    # Prepare the inputs for the model
    x_train = big_X[0:x_train.shape[0]]
    x_test = big_X[x_train.shape[0]::]

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(x_test))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, x_test, test.index

def make_data_scaled_one_hot(train_path, test_path):
    train = pd.read_csv(train_path).set_index("ID")
    test = pd.read_csv(test_path).set_index("ID")
    x_train = train.drop('target',1)
    y_train = train.target

    # Use Label str cols.
    names = list(x_train.columns.values)
    fac_columns = filter(lambda x: 'fac' in x, names)

    # Remove facVAR cols from list
    nonnumeric_columns = filter(lambda x: 'dmp' not in x, fac_columns)

    # Join the features from train and test together.
    big_X = x_train[names].append(test[names])
    # Drop all but dmp cols
    big_X = big_X.drop(nonnumeric_columns, 1)

    #dmp_cols = filter(lambda x: 'dmp' in x, fac_columns)
    #big_X = big_X.drop(list(set(names) - set(dmp_cols)), 1)

    print("Shape of Data after dropping cols:", np.shape(big_X))
    # Rescale all numeric cols to be between 0 and 1.
    names = list(big_X.columns.values)
    scaler = MinMaxScaler()
    for feature in names:
        big_X[feature] = scaler.fit_transform(big_X[feature])

    # One hot encode the categorical vars in place.
    #big_X, _ = one_hot_dataframe(big_X, cols=nonnumeric_columns,replace=True)

    # Scale all
    # le = LabelEncoder()
    # for feature in nonnumeric_columns:
    #     big_X[feature] = le.fit_transform(big_X[feature])


    # Prepare the inputs for the model
    x_train = big_X[0:x_train.shape[0]]
    x_test = big_X[x_train.shape[0]::]

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(x_test))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, x_test, test.index
