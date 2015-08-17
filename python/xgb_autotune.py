from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import auc
from xgboost import XGBClassifier
from bayesian_optimization import BayesianOptimization


def get_data():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    nunique = pd.Series([train[col].nunique() for col in train.columns], index=train.columns)
    constants = nunique[nunique < 2].index.tolist()
    train = train.drop(constants, axis=1)
    test = test.drop(constants, axis=1)

    # encode string
    strings = train.dtypes == 'object';
    strings = strings[strings].index.tolist();
    encoders = {}
    for col in strings:
        encoders[col] = preprocessing.LabelEncoder()
        train[col] = encoders[col].fit_transform(train[col])
        try:
            test[col] = encoders[col].transform(test[col])
        except:
            # lazy way to incorporate the feature only if can be encoded in the test set
            del test[col]
            del train[col]

    features = train.select_dtypes(include=['float']).columns
    features = np.setdiff1d(features, ['ID', 'target'])

    test_ids = test.ID
    y_train = train.target

    x_train = train[features]
    x_test = test[features]

    return x_train, y_train, x_test, test_ids


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent=True,
              nthread=-1,
              seed=1234):
    clf = XGBClassifier(max_depth=int(max_depth),
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        silent=silent,
                        nthread=nthread,
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        max_delta_step=max_delta_step,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        seed=seed,
                        objective="binary:logistic")

    clf.fit(train, labels, eval_metric="auc")

    return auc(clf.predict_proba(train)[:,1], labels)


if __name__ == "__main__":
    # Load data set and target values
    ts = datetime.now()
    train, labels, test, test_labels = get_data()

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (6, 10),
                                      'learning_rate': (0.5, 0.05),
                                      'n_estimators': (250, 500),
                                      'gamma': (1., 0.01),
                                      'min_child_weight': (1, 10),
                                      'max_delta_step': (0.99, 0.01),
                                      'subsample': (0.65, 0.99),
                                      'colsample_bytree': (0.8, 0.99)
                                     })

    xgboostBO.maximize()
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])

    clf = XGBClassifier(max_depth=int(xgboostBO.res['max']['max_params']['max_depth']),
                                               learning_rate=xgboostBO.res['max']['max_params']['learning_rate'],
                                               n_estimators=int(xgboostBO.res['max']['max_params']['n_estimators']),
                                               gamma=xgboostBO.res['max']['max_params']['gamma'],
                                               min_child_weight=xgboostBO.res['max']['max_params']['min_child_weight'],
                                               max_delta_step=xgboostBO.res['max']['max_params']['max_delta_step'],
                                               subsample=xgboostBO.res['max']['max_params']['subsample'],
                                               colsample_bytree=xgboostBO.res['max']['max_params']['colsample_bytree'],
                                               seed=1234,
                                               objective="binary:logistic")

    clf.fit(train, labels, eval_metric="auc")
    print('Prediction Complete')
    preds = clf.predict_proba(test.fillna(0))[:, 1]
    submission = pd.DataFrame({"ID": test_labels, "target": preds})
    submission = submission.set_index('ID')
    submission.to_csv('../xgb_autotune.csv')