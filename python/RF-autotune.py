from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from bayesian_optimization import BayesianOptimization
from make_data import make_data


def rfccv(n_estimators, min_samples_split):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               random_state=2,
                               n_jobs=-1),
                           train, train_labels, 'roc_auc', cv=2).mean()


if __name__ == "__main__":
    # Load data set and target values
    train, train_labels, test, test_labels = make_data()

    # RF
    rfcBO = BayesianOptimization(rfccv, {'n_estimators': (100, 500),
                                         'min_samples_split': (2, 5)})
    print('-' * 53)
    rfcBO.maximize()
    print('-' * 53)
    print('Final Results')
    print('RFC: %f' % rfcBO.res['max']['max_val'])

    # # MAKING SUBMISSION
    rf = cross_val_score(RFC(n_estimators=int(rfcBO['max']['max_params']['n_estimators']),
                             min_samples_split=int(rfcBO['max']['max_params']['min_samples_split']),
                             random_state=2,
                             n_jobs=-1),
                         train, test_labels, 'roc_auc', cv=2).mean()

    submission = pd.DataFrame(rf.fit(train, train_labels).predict_proba(test.fillna(0))[:, 1], index=test.index, columns=['target'])
    submission.index.name = 'ID'
    submission.to_csv('../RF-autotune.csv')