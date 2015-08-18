from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from sklearn.metrics import auc
from xgboost import XGBClassifier
from bayesian_optimization import BayesianOptimization
from make_data import make_data

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

    clf.fit(train, train_labels, eval_metric="auc")

    return auc(clf.predict_proba(train)[:,1], train_labels)


if __name__ == "__main__":
    # Load data set and target values
    train, train_labels, test, test_labels = make_data()

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

    clf.fit(train, train_labels, eval_metric="auc")
    print('Prediction Complete')
    preds = clf.predict_proba(test.fillna(0))[:, 1]
    submission = submission = pd.DataFrame(preds, index=test_labels, columns=['target'])
    submission.to_csv('../xgb_autotune.csv')