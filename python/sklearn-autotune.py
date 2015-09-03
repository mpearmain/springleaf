from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import autosklearn
import pandas as pd
from python.make_data import make_data

if __name__ == "__main__":
    # Load data set and target values
    train, train_labels, test, test_labels = make_data(train_path = "./input/train.csv", test_path="../input/test.csv")
    automl = autosklearn.AutoSklearnClassifier()
    automl.fit(train, train_labels)
    print automl.score(test, test_labels)

        print('Prediction Complete')
    preds = automl.predict_proba(test)[:, 1]
    submission = submission = pd.DataFrame(preds, index=test_labels, columns=['target'])
    submission.to_csv('../xgb_autotune.csv')