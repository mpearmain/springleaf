from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold

from make_data import make_data_scaled_one_hot


'''
    This demonstrates how to run a Keras Deep Learning model for ROC AUC score
    (local 4-fold validation) for the springleaf challenge

    The model trains in a few seconds on CPU.
'''


def float32(k):
    return np.cast['float32'](k)


def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dropout(0.1))
    model.add(Dense(input_dim, 15000, init='he_normal'))
    model.add(PReLU((15000,)))
    model.add(BatchNormalization((15000,)))
    model.add(Dropout(0.5))

    model.add(Dense(15000, 1500, init='he_normal'))
    model.add(PReLU((1500,)))
    model.add(BatchNormalization((1500,)))
    model.add(Dropout(0.5))

    model.add(Dense(1500, 1500, init='he_normal'))
    model.add(PReLU((1500,)))
    model.add(BatchNormalization((1500,)))
    model.add(Dropout(0.5))

    model.add(Dense(1500, output_dim, init='he_normal'))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="adadelta")
    return model


if __name__ == "__main__":
    # Load data set and target values
    x_train, Y, X_test, ids = \
        make_data_scaled_one_hot(train_path = "../input/xtrain_v6.csv",
                  test_path="../input/xtest_v6.csv")

    print(x_train.shape)
    #np.random.shuffle(x_train)

    x_train = x_train.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(Y).astype(np.int32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


    X_test, = X_test.astype(np.float32),
    X_test = scaler.transform(X_test)

    '''Convert class vector to binary class matrix,
    for use with categorical_crossentropy'''
    Y = np_utils.to_categorical(y)

    print('Number of classes:', len(encoder.classes_))

    input_dim = x_train.shape[1]
    output_dim = len(encoder.classes_)

    print("Validation...")

    nb_folds = 4
    kfolds = KFold(len(y), nb_folds)
    av_roc = 0.
    f = 0
    for train, valid in kfolds:
        print('---'*20)
        print('Fold', f)
        print('---'*20)
        f += 1
        X_train = x_train[train]
        X_valid = x_train[valid]
        Y_train = Y[train]
        Y_valid = Y[valid]
        y_valid = y[valid]


        print("Building model...")
        model = build_model(input_dim, output_dim)

        print("Training model...")

        model.fit(X_train, Y_train, nb_epoch=10, batch_size=128,
                  validation_data=(X_valid, Y_valid), verbose=1)
        valid_preds = model.predict_proba(X_valid, verbose=0)
        valid_preds = valid_preds[:, 1]
        roc = metrics.roc_auc_score(y_valid, valid_preds)
        print("ROC:", roc)
        av_roc += roc

    print('Average ROC:', av_roc/nb_folds)

    print("Generating submission...")

    model = build_model(input_dim, output_dim)
    model.fit(x_train, Y, nb_epoch=20, batch_size=128)

    preds = model.predict_proba(X_test, verbose=0)[:, 1]
    submission = pd.DataFrame(preds, index=ids, columns=['target'])
    submission.to_csv('Keras_BTB.csv')    
