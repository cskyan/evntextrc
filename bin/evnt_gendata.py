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
import logging
import ast
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

import bionlp.spider.pubmed as pm
import bionlp.spider.metamap as mm
import bionlp.ftslct as ftslct
import bionlp.util.io as io
import bionlp.util.sampling as sampling

import bnlpst


opts, args = {}, []
spdr = pm
SPDR_MAP = {'bnlpst':bnlpst, 'pbmd':pm}


def gen_data(scheme='trgs'):
	if (scheme == 'trgs'):
		return gen_data_trgs()
	elif (scheme == 'trg'):
		return gen_data_trg()
		
		
def gen_data_trgs():
	if (opts.local):
		train_X, train_Y = spdr.get_data(None, from_file=True, dataset='train', source=opts.year)
		dev_X, dev_Y = spdr.get_data(None, from_file=True, dataset='dev', source=opts.year)
		test_X = spdr.get_data(None, from_file=True, dataset='test', source=opts.year)
	else:
		train_docids, dev_docids, test_docids = spdr.get_docid(dataset='train'), spdr.get_docid(dataset='dev'), spdr.get_docid(dataset='test')
		train_raw_data = {
			'docids':train_docids,
			'corpus':spdr.get_corpus(train_docids, dataset='train'),
			'preprcs':spdr.get_preprcs(train_docids, dataset='train', method=opts.parser, source=opts.year),
			'evnts':spdr.get_evnts(train_docids, dataset='train', source=opts.year)
		}
		dev_raw_data = {
			'docids':dev_docids,
			'corpus':spdr.get_corpus(dev_docids, dataset='dev'),
			'preprcs':spdr.get_preprcs(dev_docids, dataset='dev', method=opts.parser, source=opts.year),
			'evnts':spdr.get_evnts(dev_docids, dataset='dev', source=opts.year)
		}
		test_raw_data = {
			'docids':test_docids,
			'corpus':spdr.get_corpus(test_docids, dataset='test'),
			'preprcs':spdr.get_preprcs(test_docids, dataset='test', method=opts.parser, source=opts.year),
			'evnts':[[]] * len(test_docids)
		}
		train_X, train_Y = spdr.get_data(train_raw_data, dataset='train', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt, parser=opts.parser, source=opts.year, store_path=mm.DATA_PATH)
		dev_X, dev_Y = spdr.get_data(dev_raw_data, dataset='dev', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt, parser=opts.parser, source=opts.year, store_path=mm.DATA_PATH)
		test_X = spdr.get_data(test_raw_data, dataset='test', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt, parser=opts.parser, source=opts.year, store_path=mm.DATA_PATH)
		print 'Training Set: X: %s, Y: %s' % (train_X.shape, train_Y.shape)
		print 'Development Set: X: %s, Y: %s' % (dev_X.shape, dev_Y.shape)
		print 'Testing Set: X: %s' % str(test_X.shape)


def gen_data_trg():
	if (opts.local):
		train_word_X, train_word_Y, train_edge_X, train_edge_Y = spdr.get_data(None, from_file=True, dataset='train', parser=opts.parser, source=opts.year)
		dev_word_X, dev_word_Y, dev_edge_X, dev_edge_Y = spdr.get_data(None, from_file=True, dataset='dev', parser=opts.parser, source=opts.year)
		test_word_X, test_rawdata = spdr.get_data(None, from_file=True, dataset='test', parser=opts.parser, source=opts.year)
	else:
		train_docids, dev_docids, test_docids = spdr.get_docid(dataset='train'), spdr.get_docid(dataset='dev'), spdr.get_docid(dataset='test')
		train_raw_data = {
			'docids':train_docids,
			'corpus':spdr.get_corpus(train_docids, dataset='train'),
			'preprcs':spdr.get_preprcs(train_docids, dataset='train', method=opts.parser, source=opts.year),
			'evnts':spdr.get_evnts(train_docids, dataset='train', source=opts.year)
		}
		dev_raw_data = {
			'docids':dev_docids,
			'corpus':spdr.get_corpus(dev_docids, dataset='dev'),
			'preprcs':spdr.get_preprcs(dev_docids, dataset='dev', method=opts.parser, source=opts.year),
			'evnts':spdr.get_evnts(dev_docids, dataset='dev', source=opts.year)
		}
		test_raw_data = {
			'docids':test_docids,
			'corpus':spdr.get_corpus(test_docids, dataset='test'),
			'preprcs':spdr.get_preprcs(test_docids, dataset='test', method=opts.parser, source=opts.year),
			'evnts':[[]] * len(test_docids)
		}
		
		train_word_X, train_word_Y, train_edge_X, train_edge_Y = spdr.get_data(train_raw_data, dataset='train', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt, source=opts.year)

		dev_word_X, dev_word_Y, dev_edge_X, dev_edge_Y = spdr.get_data(dev_raw_data, dataset='dev', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt, source=opts.year)
		test_word_X, test_rawdata = spdr.get_data(test_raw_data, dataset='test', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt, source=opts.year)
		print 'Training Set: word matrix size: %s, trigger label size: %s, edge matrix size: %s, event label size: %s.' % (train_word_X.shape, train_word_Y.shape, train_edge_X.shape, train_edge_Y.shape)
		print 'Development Set: word matrix size: %s, trigger label size: %s, edge matrix size: %s, event label size: %s.' % (dev_word_X.shape, dev_word_Y.shape, dev_edge_X.shape, dev_edge_Y.shape)
		print 'Testing Set: word matrix size: %s.' % str(test_word_X.shape)

	
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
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
	op.add_option('-l', '--local', default=False, action='store_true', dest='local', help='read data from the preprocessed data matrix file')
	op.add_option('-t', '--type', default='binary', help='feature type: binary, numeric, tfidf or mixed [default: %default]')
	op.add_option('-a', '--mindf', default='1', type='str', dest='mindf', help='lower document frequency threshold for term ignorance')
	op.add_option('-b', '--maxdf', default='1.0', type='str', dest='maxdf', help='upper document frequency threshold for term ignorance')
	op.add_option('-e', '--scheme', default='trgs', type='str', dest='scheme', help='the scheme to generate data')
	op.add_option('-i', '--input', default='bnlpst', help='input source: bnlpst or pbmd [default: %default]')
	op.add_option('-r', '--parser', default='spacy', help='the year when the data is released: spacy, stanford or bllip [default: %default]')
	op.add_option('-y', '--year', default='2016', help='the year when the data is released: 2016 or 2011 [default: %default]')
	op.add_option('-m', '--method', help='main method to run')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		exit(1)
		
	spdr = SPDR_MAP[opts.input]
	# Parse config file
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		spdr_cfg = cfgr('bionlp.spider.%s' % opts.input, 'init')
		if (len(spdr_cfg) > 0 and spdr_cfg['DATA_PATH'] is not None and os.path.exists(spdr_cfg['DATA_PATH'])):
			spdr.DATA_PATH = spdr_cfg['DATA_PATH']

	main()