from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def make_data(train_path, test_path):
    train = pd.read_csv(train_path).set_index("ID")
    test = pd.read_csv(test_path).set_index("ID")

    x_train = train.drop('target',1).fillna(0)
    y_train = train.target

    # Drop str cols.
    # facVAR0008 facVAR0226 facVAR0230 facVAR0232 facVAR0236 facVAR0404 facVAR0493

    feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
    nonnumeric_columns = ['Sex']

    # Join the features from train and test together before imputing missing values,
    # in case their distribution is slightly different
    big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

    # XGBoost doesn't (yet) handle categorical features automatically, so we need to change
    # them to columns of integer values.
    # See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
    # details and options
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        big_X[feature] = le.fit_transform(big_X[feature])


    # Prepare the inputs for the model
    train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
    test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
    train_y = train_df['Survived']

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))
    print("Shape of x_test:", np.shape(test.fillna(0)))
    print("Shape of test_index:", np.shape(test.index))

    return x_train, y_train, test.fillna(0), test.index