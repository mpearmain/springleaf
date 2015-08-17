from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC

from bayesian_optimization import BayesianOptimization


def rfccv(n_estimators, min_samples_split, max_features):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=2,
                               n_jobs=2),
                           X, y, 'roc_auc', cv=5, n_jobs=4).mean()

# PREPARE DATA
data = pd.read_csv('../input/train.csv').set_index("ID")
test = pd.read_csv('../input/test.csv').set_index("ID")

# remove constants
nunique = pd.Series([data[col].nunique() for col in data.columns], index=data.columns)
constants = nunique[nunique < 2].index.tolist()
data = data.drop(constants, axis=1)
test = test.drop(constants, axis=1)

# encode string
strings = data.dtypes == 'object'
strings = strings[strings].index.tolist()
encoders = {}
for col in strings:
    encoders[col] = preprocessing.LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])
    try:
        test[col] = encoders[col].transform(test[col])
    except:
        # lazy way to incorporate the feature only if can be encoded in the test set
        del test[col]
        del data[col]

# DATA ready
X = data.drop('target', 1).fillna(0)
y = data.target


# RF FTW :)
rfcBO = BayesianOptimization(rfccv, {'n_estimators': (10, 15),
                                     'min_samples_split': (2, 25),
                                     'max_features': (0.1, 0.999)})
print('-' * 53)
rfcBO.maximize()
print('-' * 53)
print('Final Results')
print('RFC: %f' % rfcBO.res['max']['max_val'])

# # MAKING SUBMISSION
rf = cross_val_score(RFC(n_estimators=int(rfcBO['max']['max_params']['n_estimators']),
                               min_samples_split=int(rfcBO['max']['max_params']['min_samples_split']),
                               max_features=rfcBO['max']['max_params']['max_features'],
                               random_state=2,
                               n_jobs=-1),
                           X, y, 'roc_auc', cv=5).mean()

submission = pd.DataFrame(rf.fit(X,y).predict_proba(test.fillna(0))[:,1], index=test.index, columns=['target'])
submission.index.name = 'ID'
submission.to_csv('../RF-autotune.csv')