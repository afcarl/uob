#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""Convert text into tf/idf vector in svmlight format

Usage:
python text2svmlight.py <train_filenames> <test_filenames> [args]

Args:
* `-o`, `--output` <out_train> <out_test>: output destination [required]
* `-l`, `--labels`: target labels for train files [required]
* `--filter_threshold`: lower limit of #frequency to filter [=2]
* `--lsi_topic`: #topics of LSI
* `--output_wordfreq`: if true, output #frequency of all words
"""

import argparse
import logging
import os
import sys
import gensim
import numpy as np
import pandas as pd
import re


verbose = False
logger = None


def init_logger():
    global logger
    logger = logging.getLogger('Txt2Svmlight')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def vec2dense(vec, num_terms):

    u'''Convert from sparse gensim format to dense list of numbers'''
    return list(gensim.matutils.corpus2dense([vec], num_terms=num_terms).T[0])


def output_lsimodel(lsimodel, bow, filename, n_topic=2):
    keys = bow.keys()
    with open(filename, 'w') as f:
        for key in sorted(keys):
            f.write('{key},'.format(key=key))
            result = [0 for i in range(n_topic)]
            for val in lsimodel[bow[key]]:
                result[val[0]] = val[1]
            f.write('{vals}\n'.format(vals=','.join(map(str, result))))

def load_file_as_list(filename):
    with open(filename) as f:
        l = f.read().splitlines()
    return l


def load_files(filenames, datadir):
    docs = {}
    for filename in filenames:
        path = os.path.join(datadir, filename)
        docs[filename] = open(path).read().strip()
    return docs


def make_corpus(docs, stopwords=set([])):
    corpus = {}
    for key, val in docs.items():
        corpus[key] = [word for word in val.split()
                       if word not in stopwords]
        for i in range(len(corpus[key])):
            if not corpus[key][i][1:].isdigit():
                r = re.compile('[0-9]+[a-zA-Z]')
                m = r.search(corpus[key][i][1:])
                corpus[key][i] = corpus[key][i][0] + m.group(0)[:-1]
    return corpus


def main(args):
    global verbose
    verbose = args.verbose

    # Load in corpus
    train_filenames = load_file_as_list(args.train_filenames)
    test_filenames = load_file_as_list(args.test_filenames)
    train_dir = os.path.join(os.path.dirname(args.train_filenames), 'train')
    test_dir = os.path.join(os.path.dirname(args.test_filenames), 'test')

    if verbose: logger.info('Loading train files')
    train_docs = load_files(train_filenames, train_dir)
    n_train = len(train_docs)
    train_names = train_docs.keys()

    if verbose: logger.info('Loading test files')
    test_docs = load_files(test_filenames, test_dir)
    n_test = len(test_docs)
    test_names = test_docs.keys()

    # Split on spaces
    train_corpus = make_corpus(train_docs)
    test_corpus = make_corpus(test_docs)

    # Build a dictionary
    if verbose: logger.info('Building a dictionary')
    dct = gensim.corpora.Dictionary(train_corpus.values() + test_corpus.values())
    unfiltered = dct.token2id.keys()
    if args.filter_threshold:
        if verbose: logger.info('Filtering with freq >= {}'.format(args.filter_threshold))
        dct.filter_extremes(no_below=args.filter_threshold)
    filtered = dct.token2id.keys()
    if args.filter_threshold:
        filtered_out = set(unfiltered) - set(filtered)
        if verbose: logger.info('Filtered out {} words from {}'.format(len(filtered_out), len(unfiltered)))
    if args.output_wordfreq:
        dct.save_as_text('word_freq.tsv', sort_by_word=False)

    # Build bag of words vectors
    if verbose: logger.info('Building bag of words vectors')
    train_bow = {name: dct.doc2bow(train_corpus[name])
                 for name in train_names}
    test_bow = {name: dct.doc2bow(test_corpus[name])
                 for name in test_names}

    # TF-IDF model
    if verbose: logger.info('TF-IDF Model')
    tfidf_model = gensim.models.TfidfModel(dictionary=dct)
    train_tfidf = {name: tfidf_model[train_bow[name]]
                   for name in train_names}
    test_tfidf = {name: tfidf_model[test_bow[name]]
                   for name in test_names}

    # LSI
    if not args.lsi_topic is None:
        if verbose: logger.info('LSI Model')

        train_lsi = {}
        test_lsi = {}
        num_topics = args.lsi_topic
        lsi_model = gensim.models.LsiModel(train_tfidf.values() + test_tfidf.values(),
                                           num_topics=num_topics)

        for name in train_names:
            vec = train_tfidf[name]
            train_lsi[name] = lsi_model[vec]
        for name in test_names:
            vec = test_tfidf[name]
            test_lsi[name] = lsi_model[vec]
    else:
        train_lsi = train_tfidf
        test_lsi = test_tfidf

    # Output
    if verbose: logger.info('Writing to {}'.format(args.output[0]))
    idx, train_bow = zip(*sorted(train_lsi.items(),
                                 key=lambda t:int(t[0].split('.')[0])))

    label_dat = pd.read_csv(args.label,
                            names=['id','cls'], header=None, index_col=0)
    labels = list(label_dat.ix[idx, 'cls'].values)
    gensim.corpora.SvmLightCorpus.serialize(args.output[0], train_bow, labels=labels)
    if verbose: logger.info('Writing to {}'.format(args.output[1]))
    idx, test_bow = zip(*sorted(test_lsi.items(),
                                 key=lambda t:int(t[0].split('.')[0])))

    gensim.corpora.SvmLightCorpus.serialize(args.output[1], test_bow,
                                            labels=[0 for i in range(n_test)])
    return 0

if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('train_filenames')
    parser.add_argument('test_filenames')
    parser.add_argument('-l', '--label', required=True)
    parser.add_argument('--lsi_topic', type=int)
    parser.add_argument('--filter_threshold', type=int, default=int(2))
    parser.add_argument('--output_wordfreq', action='store_true')
    parser.add_argument('-o', '--output', required=True, nargs=2)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
