__author__ = 'michael.pearmain'

from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from bayesian_optimization import BayesianOptimization
from make_data import make_data
from ftrl-benchmark


def ftrlcv(alpha, beta, L1, L2, epoch,):
    D = 2 ** 24
    interaction = False
    learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
    # start training
    for e in range(epoch):
        for t,  x, y in data(train, D):  # data is a generator
            p = learner.predict(x)
            learner.update(x, p, y)




if __name__ == "__main__":
    # Load data set and target values
    train, train_labels, test, test_labels = make_data(train_path = "../input/train.csv", test_path="../input/test.csv")

    ftrlBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (7, 20),
                                      'learning_rate': (0.45, 0.01),
                                      'n_estimators': (100, 500),
                                      'gamma': (1., 0.1),
                                      'min_child_weight': (2, 15),
                                      'max_delta_step': (0.6, 0.4),
                                      'subsample': (0.7, 0.9),
                                      'colsample_bytree': (0.7, 0.9)
                                     })

    ftrlBO.maximize(init_points=5, restarts=50, n_iter=25)
    print('-' * 53)

    print('Final Results')
    print('FTRL: %f' % ftrlBO.res['max']['max_val'])


    # Build and Run on the full data set and the validation set for ensembling later.

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
    preds = clf.predict_proba(test)[:, 1]
    submission = submission = pd.DataFrame(preds, index=test_labels, columns=['target'])
    submission.to_csv('../output/xgb_autotune.csv')