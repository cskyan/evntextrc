#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: evnt_gendata.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-07-05 14:41:59
###########################################################################
#

import os
import sys
import logging
import ast
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

from bionlp.spider import w2v
import bionlp.spider.pubmed as pm
import bionlp.spider.metamap as mm
import bionlp.ftslct as ftslct
import bionlp.util.io as io
import bionlp.util.sampling as sampling

import bnlpst


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'bnlpst':bnlpst, 'pbmd':pm}
SC=';;'

opts, args = {}, []
cfgr = None
spdr = pm


def stat_dataset(dataset, ds_name):
	preprcs = dataset['preprcs']
	word_count = []
	for sent_bndry, words, annots, gddfs, coref in preprcs:
		word_count.append(len(words['str']))
	print 'The average word number in each document of %s dataset is: %i' % (ds_name, np.mean(word_count))
	io.write_npz(np.array(word_count), fpath='%s_wordcount' % ds_name)


def gen_data(scheme='trgs'):
	if (scheme == 'trgs'):
		return gen_data_trgs()
	elif (scheme == 'trg'):
		return gen_data_trg()
	elif (scheme == 'jointee'):
		return gen_data_jointee()
	elif (scheme == 'cbow'):
		return gen_data_cbow()


def gen_data_trgs():
	if (opts.local):
		train_X, train_Y = spdr.get_data(None, method='trigger', scheme='trgs', from_file=True, dataset='train', year=opts.year, task=opts.task)
		dev_X, dev_Y = spdr.get_data(None, method='trigger', scheme='trgs', from_file=True, dataset='dev', year=opts.year, task=opts.task)
		test_X = spdr.get_data(None, method='trigger', scheme='trgs', from_file=True, dataset='test', year=opts.year, task=opts.task)
	else:
		train_docids, dev_docids, test_docids = spdr.get_docid(dataset='train', year=opts.year, task=opts.task), spdr.get_docid(dataset='dev', year=opts.year, task=opts.task), spdr.get_docid(dataset='test', year=opts.year, task=opts.task)
		train_raw_data = {
			'docids':train_docids,
			'corpus':spdr.get_corpus(train_docids, dataset='train', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(train_docids, dataset='train', year=opts.year, task=opts.task, method=opts.parser),
			'evnts':spdr.get_evnts(train_docids, dataset='train', year=opts.year, task=opts.task)
		}
		dev_raw_data = {
			'docids':dev_docids,
			'corpus':spdr.get_corpus(dev_docids, dataset='dev', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(dev_docids, dataset='dev', year=opts.year, task=opts.task, method=opts.parser),
			'evnts':spdr.get_evnts(dev_docids, dataset='dev', year=opts.year, task=opts.task)
		}
		test_raw_data = {
			'docids':test_docids,
			'corpus':spdr.get_corpus(test_docids, dataset='test', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(test_docids, dataset='test', year=opts.year, task=opts.task, method=opts.parser),
			'evnts':[[]] * len(test_docids)
		}
		train_X, train_Y = spdr.get_data(train_raw_data, method='trigger', scheme='trgs', dataset='train', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), parser=opts.parser, store_path=mm.DATA_PATH)
		dev_X, dev_Y = spdr.get_data(dev_raw_data, method='trigger', scheme='trgs', dataset='dev', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), parser=opts.parser, store_path=mm.DATA_PATH)
		test_X = spdr.get_data(test_raw_data, method='trigger', scheme='trgs', dataset='test', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), parser=opts.parser, store_path=mm.DATA_PATH)
		print 'Training Set: X: %s, Y: %s' % (train_X.shape, train_Y.shape)
		print 'Development Set: X: %s, Y: %s' % (dev_X.shape, dev_Y.shape)
		print 'Testing Set: X: %s' % str(test_X.shape)


def gen_data_trg():
	if (opts.local):
		train_word_X, train_word_Y, train_edge_X, train_edge_Y = spdr.get_data(None, method='trigger', scheme='trg', from_file=True, dataset='train', year=opts.year, task=opts.task, parser=opts.parser)
		dev_word_X, dev_word_Y, dev_edge_X, dev_edge_Y = spdr.get_data(None, method='trigger', scheme='trg', from_file=True, dataset='dev', year=opts.year, task=opts.task, parser=opts.parser)
		test_word_X, test_rawdata = spdr.get_data(None, method='trigger', scheme='trg', from_file=True, dataset='test', year=opts.year, task=opts.task, parser=opts.parser)
	else:
		train_docids, dev_docids, test_docids = spdr.get_docid(dataset='train', year=opts.year, task=opts.task), spdr.get_docid(dataset='dev', year=opts.year, task=opts.task), spdr.get_docid(dataset='test', year=opts.year, task=opts.task)
		train_raw_data = {
			'docids':train_docids,
			'corpus':spdr.get_corpus(train_docids, dataset='train', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(train_docids, dataset='train', year=opts.year, task=opts.task, method=opts.parser),
			'evnts':spdr.get_evnts(train_docids, dataset='train', year=opts.year, task=opts.task)
		}
		dev_raw_data = {
			'docids':dev_docids,
			'corpus':spdr.get_corpus(dev_docids, dataset='dev', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(dev_docids, dataset='dev', year=opts.year, task=opts.task, method=opts.parser),
			'evnts':spdr.get_evnts(dev_docids, dataset='dev', year=opts.year, task=opts.task)
		}
		test_raw_data = {
			'docids':test_docids,
			'corpus':spdr.get_corpus(test_docids, dataset='test', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(test_docids, dataset='test', year=opts.year, task=opts.task, method=opts.parser),
			'evnts':[[]] * len(test_docids)
		}

		train_word_X, train_word_Y, train_edge_X, train_edge_Y = spdr.get_data(train_raw_data, method='trigger', scheme='trg', dataset='train', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), parser=opts.parser)

		dev_word_X, dev_word_Y, dev_edge_X, dev_edge_Y = spdr.get_data(dev_raw_data, method='trigger', scheme='trg', dataset='dev', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), parser=opts.parser)
		test_word_X, test_rawdata = spdr.get_data(test_raw_data, method='trigger', scheme='trg', dataset='test', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), parser=opts.parser)
		print 'Training Set: word matrix size: %s, trigger label size: %s, edge matrix size: %s, event label size: %s.' % (train_word_X.shape, train_word_Y.shape, train_edge_X.shape, train_edge_Y.shape)
		print 'Development Set: word matrix size: %s, trigger label size: %s, edge matrix size: %s, event label size: %s.' % (dev_word_X.shape, dev_word_Y.shape, dev_edge_X.shape, dev_edge_Y.shape)
		print 'Testing Set: word matrix size: %s.' % str(test_word_X.shape)


def gen_data_jointee():
	if (opts.local):
		train_Xs, train_Y = spdr.get_data(None, method='jointee', from_file=True, dataset='train', year=opts.year, task=opts.task)
		dev_Xs, dev_Y = spdr.get_data(None, method='jointee', from_file=True, dataset='dev', year=opts.year, task=opts.task)
		test_Xs = spdr.get_data(None, method='jointee', from_file=True, dataset='test', year=opts.year, task=opts.task)
	else:
		train_docids, dev_docids, test_docids = spdr.get_docid(dataset='train', year=opts.year, task=opts.task), spdr.get_docid(dataset='dev', year=opts.year, task=opts.task), spdr.get_docid(dataset='test', year=opts.year, task=opts.task)
		train_raw_data = {
			'docids':train_docids,
			'corpus':spdr.get_corpus(train_docids, dataset='train', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(train_docids, dataset='train', year=opts.year, task=opts.task, method=opts.parser, disable=['parse'] if opts.year==2011 else []),
			'evnts':spdr.get_evnts(train_docids, dataset='train', year=opts.year, task=opts.task)
		}
		dev_raw_data = {
			'docids':dev_docids,
			'corpus':spdr.get_corpus(dev_docids, dataset='dev', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(dev_docids, dataset='dev', year=opts.year, task=opts.task, method=opts.parser, disable=['parse'] if opts.year==2011 else []),
			'evnts':spdr.get_evnts(dev_docids, dataset='dev', year=opts.year, task=opts.task)
		}
		test_raw_data = {
			'docids':test_docids,
			'corpus':spdr.get_corpus(test_docids, dataset='test', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(test_docids, dataset='test', year=opts.year, task=opts.task, method=opts.parser, disable=['parse'] if opts.year==2011 else []),
			'evnts':[[]] * len(test_docids)
		}
		kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
		train_Xs, train_Y = spdr.get_data(train_raw_data, method='jointee', ret_field='all', dataset='train', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, w2v_path=kwargs.setdefault('wordvec', None), window_size=kwargs.setdefault('window_size', 10), maxlen=kwargs.setdefault('maxlen', None), npg_ratio=kwargs.setdefault('npg_ratio', 1.0))
		dev_Xs, dev_Y = spdr.get_data(dev_raw_data, method='jointee', ret_field='all', dataset='dev', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, w2v_path=kwargs.setdefault('wordvec', None), window_size=kwargs.setdefault('window_size', 10), maxlen=kwargs.setdefault('maxlen', None), npg_ratio=kwargs.setdefault('npg_ratio', 1.0))
		test_Xs, test_rawdata = spdr.get_data(test_raw_data, method='jointee', ret_field='all', dataset='test', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, w2v_path=kwargs.setdefault('wordvec', None), window_size=kwargs.setdefault('window_size', 10), maxlen=kwargs.setdefault('maxlen', None), npg_ratio=kwargs.setdefault('npg_ratio', 1.0))
		print 'Training Set: X: %s, Y: %s' % (', '.join([str(x.shape) for x in train_Xs]), ', '.join([str(y.shape) for y in train_Y]))
		print 'Development Set: X: %s, Y: %s' % (', '.join([str(x.shape) for x in dev_Xs]), ', '.join([str(y.shape) for y in dev_Y]))
		print 'Testing Set: X: %s' % ', '.join([str(x.shape) for x in test_Xs])


def gen_data_cbow():
	if (opts.local):
		train_Xs, train_Y = spdr.get_data(None, method='cbow', from_file=True, dataset='train', year=opts.year, task=opts.task)
		dev_Xs, dev_Y = spdr.get_data(None, method='cbow', from_file=True, dataset='dev', year=opts.year, task=opts.task)
		test_Xs = spdr.get_data(None, method='cbow', from_file=True, dataset='test', year=opts.year, task=opts.task)
	else:
		train_docids, dev_docids, test_docids = spdr.get_docid(dataset='train', year=opts.year, task=opts.task), spdr.get_docid(dataset='dev', year=opts.year, task=opts.task), spdr.get_docid(dataset='test', year=opts.year, task=opts.task)
		train_raw_data = {
			'docids':train_docids,
			'corpus':spdr.get_corpus(train_docids, dataset='train', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(train_docids, dataset='train', year=opts.year, task=opts.task, method=opts.parser, disable=['parse'] if opts.year==2011 else []),
			'evnts':spdr.get_evnts(train_docids, dataset='train', year=opts.year, task=opts.task)
		}
		stat_dataset(train_raw_data, 'train')
		dev_raw_data = {
			'docids':dev_docids,
			'corpus':spdr.get_corpus(dev_docids, dataset='dev', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(dev_docids, dataset='dev', year=opts.year, task=opts.task, method=opts.parser, disable=['parse'] if opts.year==2011 else []),
			'evnts':spdr.get_evnts(dev_docids, dataset='dev', year=opts.year, task=opts.task)
		}
		stat_dataset(dev_raw_data, 'dev')
		test_raw_data = {
			'docids':test_docids,
			'corpus':spdr.get_corpus(test_docids, dataset='test', year=opts.year, task=opts.task),
			'preprcs':spdr.get_preprcs(test_docids, dataset='test', year=opts.year, task=opts.task, method=opts.parser, disable=['parse'] if opts.year==2011 else []),
			'evnts':[[]] * len(test_docids)
		}
		stat_dataset(test_raw_data, 'test')
		kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
		train_Xs, train_Y = spdr.get_data(train_raw_data, method='cbow', ret_field='all', dataset='train', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, w2v_path=kwargs.setdefault('wordvec', None), cw2v_path=kwargs.setdefault('cncptvec', None), window_size=kwargs.setdefault('wsize', 10), maxlen=kwargs.setdefault('maxlen', None), npg_ratio=kwargs.setdefault('npg_ratio', 1.0), concept_embed=opts.concept)
		dev_Xs, dev_Y = spdr.get_data(dev_raw_data, method='cbow', ret_field='all', dataset='dev', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, w2v_path=kwargs.setdefault('wordvec', None), cw2v_path=kwargs.setdefault('cncptvec', None), window_size=kwargs.setdefault('wsize', 10), maxlen=kwargs.setdefault('maxlen', None), npg_ratio=kwargs.setdefault('npg_ratio', 1.0), concept_embed=opts.concept)
		test_Xs, test_rawdata = spdr.get_data(test_raw_data, method='cbow', ret_field='all', dataset='test', year=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt, w2v_path=kwargs.setdefault('wordvec', None), cw2v_path=kwargs.setdefault('cncptvec', None), window_size=kwargs.setdefault('wsize', 10), maxlen=kwargs.setdefault('maxlen', None), npg_ratio=kwargs.setdefault('npg_ratio', 1.0), concept_embed=opts.concept)
		print 'Training Set: X: %s, Y: %s' % (', '.join([str(x.shape) for x in train_Xs]), ', '.join([str(y.shape) for y in train_Y]))
		print 'Development Set: X: %s, Y: %s' % (', '.join([str(x.shape) for x in dev_Xs]), ', '.join([str(y.shape) for y in dev_Y]))
		print 'Testing Set: X: %s' % ', '.join([str(x.shape) for x in test_Xs])


def main():
	global spdr
	spdr = SPDR_MAP[opts.input]
	if (opts.method is None):
		return
	elif (opts.method == 'gen'):
		gen_data(scheme=opts.scheme)


if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-p', '--pid', action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
	op.add_option('-f', '--fmt', default='h5', help='data stored format: csv, npz, or h5 [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
	op.add_option('-c', '--cfg', help='config string used in the plot function, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
	op.add_option('-l', '--local', default=False, action='store_true', dest='local', help='read data from the preprocessed data matrix file')
	op.add_option('-t', '--type', default='binary', help='feature type: binary, numeric, tfidf or mixed [default: %default]')
	op.add_option('-a', '--mindf', default='1', type='str', dest='mindf', help='lower document frequency threshold for term ignorance')
	op.add_option('-b', '--maxdf', default='1.0', type='str', dest='maxdf', help='upper document frequency threshold for term ignorance')
	op.add_option('-e', '--scheme', default='cbow', type='str', dest='scheme', help='the scheme to generate data')
	op.add_option('-w', '--wsize', default=10, action='store', type='int', dest='wsize', help='indicate the window size for bi-directional word stream extraction')
	op.add_option('--concept', default=False, action='store_true', dest='concept', help='indicate whether use concept embedding')
	op.add_option('-i', '--input', default='bnlpst', help='input source: bnlpst or pbmd [default: %default]')
	op.add_option('-r', '--parser', default='spacy', help='the year when the data is released: spacy, stanford or bllip [default: %default]')
	op.add_option('-y', '--year', default='2016', help='the year when the data is released: 2016 or 2011 [default: %default]')
	op.add_option('-u', '--task', default='bb', help='the year when the data is released: 2016 or 2011 [default: %default]')
	op.add_option('-m', '--method', help='main method to run')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		sys.exit(1)

	spdr = SPDR_MAP[opts.input]
	# Parse config file
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		spdr_cfg = cfgr('bionlp.spider.%s' % opts.input, 'init')
		if (len(spdr_cfg) > 0 and spdr_cfg['DATA_PATH'] is not None and os.path.exists(spdr_cfg['DATA_PATH'])):
			spdr.DATA_PATH = spdr_cfg['DATA_PATH']
		w2v_cfg = cfgr('bionlp.spider.w2v', 'init')
		if (len(w2v_cfg) > 0 and w2v_cfg['DATA_PATH'] is not None and w2v_cfg['W2V_MODEL'] is not None and os.path.exists(os.path.join(w2v_cfg['DATA_PATH'], w2v_cfg['W2V_MODEL']))):
			w2v.DATA_PATH = w2v_cfg['DATA_PATH']
			w2v.W2V_MODEL = w2v_cfg['W2V_MODEL']


	main()
