#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
"""

import argparse
import logging
import sys
import gensim
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

verbose = False
logger = None
   

def init_logger():
    global logger
    logger = logging.getLogger('SVC')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def load_file(filename):
    X, y = load_svmlight_file(filename)
    return X, y


def main(args):
    global verbose
    verbose = args.verbose

    # Load files
    if verbose: logger.info('Loading {}'.format(args.train_file))
    train_X, train_y = load_file(args.train_file)
    if verbose: logger.info('Loading {}'.format(args.test_file))
    test_X, test_y = load_file(args.test_file)

    # To dense
    train_X = train_X.toarray()
    test_X = test_X.toarray()

    # # Codes for Grid Search
    # params = [
    #     {'C': [2**i for i in range(0, 3, 1)], 'gamma': [2**i for i in np.arange(-1, 2, 0.5)], 'kernel': ['rbf']},
    # ]
    # method = SVC(cache_size=1024, probability=True, random_state=1)
    # gscv = GridSearchCV(method, params, scoring='roc_auc', verbose=verbose, n_jobs=9)
    # gscv.fit(train_X, train_y)

    # if verbose:
    #     for params, mean_score, all_scores in gscv.grid_scores_:
    #         logger.info('{:.3f} (+/- {:.3f}) for {}'.format(mean_score, all_scores.std() / 2, params))
    #     logger.info('params:{params}'.format(params=gscv.best_params_))
    #     logger.info('score:{params}'.format(params=gscv.best_score_))
    # pred = gscv.best_estimator_.predict_proba(test_X)

    # Best parameter for the competition data
    method = SVC(kernel='rbf', C=4, gamma=0.5, cache_size=1024, probability=True, random_state=1)
    method.fit(train_X, train_y)
    pred = method.predict_proba(test_X)

    np.savetxt(args.output, pred[:, 1], fmt='%.6f')
    if verbose: logger.info('Wrote preds to {file}'.format(file=args.output))

    return 0

if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
