from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from make_data import make_data

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

if __name__ == "__main__":
    # Load data set and target values
    print("Loading Data sets")
    train, train_labels, test, test_labels = \
        make_data(train_path="../input/xtrain_v1_r2.csv", test_path="../input/xtest_v1_r2.csv")

    num_classes = 2
    num_features = train.shape[1]

    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout', DropoutLayer),
               ('dense1', DenseLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,

                     input_shape=(None, num_features),
                     dense0_num_units=200,
                     dropout_p=0.5,
                     dense1_num_units=200,
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,

                     update=nesterov_momentum,
                     update_learning_rate=0.01,
                     update_momentum=0.9,

                     train_split=0.1,
                     verbose=1,
                     max_epochs=20)

    net0.fit(train, train_labels)
    print('Prediction Complete')
    preds = net0.predict_proba(test)[:, 1]
    submission = pd.DataFrame(preds, index=test_labels, columns=['target'])
    submission.to_csv('../output/NN_autotune.csv')