from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from theano import shared
from theano import tensor as T
from theano.tensor.nnet import sigmoid

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def float32(k):
    return np.cast['float32'](k)

def load_train_data(path):
    print("Loading Train Data")
    df = pd.read_csv(path)
    labels = df.target

    df = df.drop('target',1)
    df = df.drop('ID',1)
    X = df.values.copy()

    np.random.shuffle(X)

    X = X.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    print("Loading Test Data")
    df = pd.read_csv(path)
    ids = df.ID.astype(str)

    df = df.drop('ID',1)
    X = df.values.copy()

    X, = X.astype(np.float32),
    X = scaler.transform(X)
    return X, ids

if __name__ == "__main__":

    # Load data set and target values

    X, y, encoder, scaler = load_train_data("../input/train.csv")
    X_test, ids = load_test_data("../input/test.csv", scaler)
    print('Number of classes:', len(encoder.classes_))
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]

    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense2', DenseLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,

                     input_shape=(None, num_features),
                     dense0_num_units=313,
                     dropout0_p=0.1,
                     dense1_num_units=313,
                     dropout1_p=0.1,
                     dense2_num_units=39,

                     output_num_units=num_classes,
                     output_nonlinearity=softmax,

                     update=nesterov_momentum,
                     update_learning_rate=shared(float32(0.3)),
                     update_momentum=shared(float32(0.8)),
                     on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.3, stop=0.1),
                        AdjustVariable('update_momentum', start=0.8, stop=0.999),
                        EarlyStopping(patience=10),
                     ],

                     train_split=TrainSplit(0.1),
                     verbose=1,
                     max_epochs=10)

    net0.fit(X, y)
    print('Prediction Complete')
    preds = net0.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(preds, index=ids, columns=['target'])
    submission.to_csv('../output/NN_autotune.csv')