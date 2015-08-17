from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import autosklearn
import numpy as np
import pandas as pd
from make_data import make_data

if __name__ == "__main__":
    # Load data set and target values
    train, train_labels, test, test_labels = make_data()
    automl = autosklearn.AutoSklearnClassifier()
    automl.fit(train, train_labels)
    print automl.score(test, test_labels)

    submission = pd.DataFrame(automl.fit(train, train_labels).predict_proba(test.fillna(0))[:, 1], index=test.index, columns=['target'])
    submission.index.name = 'ID'
    submission.to_csv('../sklearn-autotune.csv')
