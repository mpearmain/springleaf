from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer


''' Below are two functions for creating data,
 make_data() converts data for using with xgboost and other sklearn functions
 make_data_keras() is specific to keras NN
'''

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

def make_data_keras(train_path, test_path):
    # Load the data and set the index for the pandas data
    train = pd.read_csv(train_path).set_index("ID")
    test = pd.read_csv(test_path).set_index("ID")
    # Drop the target var
    x_train = train.drop('target',1)
    y_train = train.target
    # Clean up train
    del train

    names = x_train.columns.values
    # Join the features from train and test together.
    big_X = x_train[names].append(test[names])
    print("Shape of Data before dropping cols:", np.shape(big_X))

    # Remove facVAR cols from data but keep the facVARdmp vars to model them.
    # One-hot encoding the categorical data produces huge data bloat
    # We have continuous vars to use instead.
    nonnumeric_columns = filter(lambda x: 'fac' in x, names)
    nonnumeric_columns = filter(lambda x: 'dmp' not in x, nonnumeric_columns)

    # XGBoost doesn't (yet) handle categorical features automatically,
    # so we need to change them to columns of integer values.
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        big_X[feature] = le.fit_transform(big_X[feature])

    # Get the categorical values
    big_X_categorical_values = big_X[nonnumeric_columns]

    # values appearing less than min_obs are grouped into one dummy variable.
    big_X_categorical_values = [dict(r.iteritems()) for _, r in big_X_categorical_values.iterrows()]
    train_fea = DictVectorizer(sparse=False).fit_transform(big_X_categorical_values).toarray()

    print(train_fea.shape)
    train_fea = pd.DataFrame(train_fea)


    big_X = big_X.drop(nonnumeric_columns, 1)

    # Rescale all cols.
    big_X = big_X.astype(np.float32)
    scaler = StandardScaler()
    big_X = scaler.fit_transform(big_X)

    # Add the one hot encodings to the big_X matrix.
    big_X = pd.concat([big_X, train_fea], axis=1)
    print("Shape of Data after adding one hot cols:", np.shape(big_X))
    # Prepare the inputs for the model
    x_train = big_X[0:x_train.shape[0]]
    x_test = big_X[x_train.shape[0]::]

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(x_test))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, x_test, test.index