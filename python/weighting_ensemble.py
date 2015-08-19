__author__ = 'michael.pearmain'
from __future__ import division

from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score as auc
import numpy
import pandas as pd

def weighted_auc(weights, clicks, pctrs):
        """
        :param clicks: vector of real click/noclick events
        :param pctrs: 2D matrix of predicted CTR for each model across all
        requests
        :param weights: vector of weights of individual models
        :return: final auc
        """
        assert len(pctrs[0]) == len(weights), 'wrong dimension of weights-pctrs'
        weighted_pctr = []
        for row in pctrs:
            w_sum = 0
            for idx, weight in enumerate(weights):
                w_sum += row[idx] * weight
            weighted_pctr.append([1-w_sum, w_sum])
        print
        print "Sklearn AUC =", auc(clicks, weighted_pctr), "Weights = ", weights
        return auc(clicks, weighted_pctr)


## To use the scipy.minimize function we first save model probabilities like so:
# pctrs=[p1,p2,p3,p4]
ensemble_data = pd.read_csv("validationPCTR.csv", sep=" ")
clicks = ensemble_data["click"].values.tolist()
clicks = [float(i) for i in clicks]
pctrs = ensemble_data.iloc[:, 2:].values

restarts = 10
ei_max = 0

## init_weights is the initial guess for the minimum of function 'fun'
## This initial guess is that all weights are equal
init_weights = [1./(pctrs.shape[1])] * pctrs.shape[1]
# ## This sets the bounds on the weights, between 0 and 1b
bnds = tuple((0,1) for w in init_weights)
## This sets the constraints on the weights, they must sum to 1
## Or, in other words, 1 - sum(w) = 0
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})


for i in range(restarts):

    # Find the minimum auc
    weights = minimize(
        weighted_auc,
        init_weights,
        (clicks, pctrs),
        method='SLSQP',
        bounds=bnds,
        constraints=cons,
        options={'ftol': 1e-8}
    )

    print -weights.fun
    print weights.x
    # Store it if better than previous minimum.
    if -weights.fun <= ei_max:
        x_max = weights.x
        ei_max = -weights.fun

    init_weights = numpy.random.dirichlet(numpy.ones(pctrs.shape[1]),size=1)

## As a sanity check, make sure the weights do in fact sum to 1
print "Optimal solution auc = ", ei_max
print "Optimal weights = ", x_max
print "Weights sum to %0.4f:" % sum(x_max)


