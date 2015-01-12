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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Normalizer, StandardScaler

verbose = False
logger = None
   

def init_logger():
    global logger
    logger = logging.getLogger('GBC')
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

    # # Codes for Grid Search
    # params = [
    #     {'n_estimators': [50000], 'learning_rate': [2**i for i in np.arange(-10, -9, .25)], 'max_features': ['log2',], 'max_depth': [7,]},
    # ]
    # method = GradientBoostingClassifier(random_state=1, verbose=1)
    # gscv = GridSearchCV(method, params, scoring='roc_auc', verbose=verbose, n_jobs=5)
    # gscv.fit(train_X.toarray(), train_y)
    # if verbose:
    #     for params, mean_score, all_scores in gscv.grid_scores_:
    #         logger.info('{:.6f} (+/- {:.6f}) for {}'.format(mean_score, all_scores.std() / 2, params))
    #     logger.info('params:{params}'.format(params=gscv.best_params_))
    #     logger.info('score:{params}'.format(params=gscv.best_score_))
    # pred = gscv.best_estimator_.predict_proba(test_X.toarray())

    # Best parameters for the competition data
    method = GradientBoostingClassifier(n_estimators=50000, learning_rate=2**(-9,5),
                                        max_features='log2', max_depth=7
                                        random_state=1, verbose=1)
    method.fit(train_X.toarray(), train_y)
    pred = method.predict_proba(test_X.toarray())

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
