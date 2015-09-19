from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from xgboost import XGBClassifier
from bayesian_optimization import BayesianOptimization
from make_data import make_data
from sklearn.cross_validation import train_test_split

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

    # Run Kfolds on the data model to stop over-fitting
    X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                          train_labels,
                                                          test_size=0.1,
                                                          random_state=seed)
    xgb_model = clf.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
    y_pred = xgb_model.predict_proba(X_valid)[:,1]

    return auc(y_valid, y_pred)

if __name__ == "__main__":
    # Load data set and target values
    train, train_labels, test, test_labels = \
        make_data(train_path = "../input/xtrain_v6.csv", test_path="../input/xtest_v6.csv")

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (8, 30),
                                      'learning_rate': (0.8, 0.1),
                                      'n_estimators': (250, 1500),
                                      'gamma': (1., 0.01),
                                      'min_child_weight': (2, 20),
                                      'max_delta_step': (0., 0.3),
                                      'subsample': (0.7, 0.85),
                                      'colsample_bytree': (0.7, 0.85)
                                     })

    xgboostBO.maximize(init_points=7, restarts=50, n_iter=30)
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])


    # Build and Run on the full data set K-fold times for bagging

    seeds = [1234, 5434, 87897, 123125, 88888]
    for seed_bag in seeds:
        X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                              train_labels,
                                                              test_size=0.1,
                                                              random_state=seed_bag)
        clf = XGBClassifier(max_depth=int(xgboostBO.res['max']['max_params']['max_depth']),
                                               learning_rate=xgboostBO.res['max']['max_params']['learning_rate'],
                                               n_estimators=int(xgboostBO.res['max']['max_params']['n_estimators']),
                                               gamma=xgboostBO.res['max']['max_params']['gamma'],
                                               min_child_weight=xgboostBO.res['max']['max_params']['min_child_weight'],
                                               max_delta_step=xgboostBO.res['max']['max_params']['max_delta_step'],
                                               subsample=xgboostBO.res['max']['max_params']['subsample'],
                                               colsample_bytree=xgboostBO.res['max']['max_params']['colsample_bytree'],
                                               seed=seed_bag,
                                               objective="binary:logistic")

        clf.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
        print('Prediction Complete')
        preds = clf.predict_proba(test)[:, 1]
        submission = submission = pd.DataFrame(preds, index=test_labels, columns=['target'])
        outfile_seed = '../output/xgb_autotune' + str(seed_bag) + '.csv'
        submission.to_csv(outfile_seed)