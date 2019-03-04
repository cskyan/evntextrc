#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: evnt_extrc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-07-15 14:38:16
###########################################################################
#

import os
import sys
import logging
import itertools
from optparse import OptionParser
from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, SelectFpr, SelectFromModel, chi2, f_classif
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier, LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso
from sklearn.svm import SVC, LinearSVC
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from bionlp import ftslct, txtclf
from bionlp.model import kerasext, vecomnet
from bionlp.util import fs, io, func
from bionlp.spider import w2v
import bionlp.util.math as imath
import bionlp.spider.pubmed as pm

import bnlpst


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'bnlpst':bnlpst, 'pbmd':pm}
FILT_NAMES, CLF_NAMES, PL_NAMES = [[] for i in range(3)]
TRG_FILT_NAMES, TRG_CLF_NAMES, EDGE_FILT_NAMES, EDGE_CLF_NAMES, TRG_PL_NAMES, EDGE_PL_NAMES = [[] for i in range(6)]
PL_SET = set([])
TRG_PL_SET, EDGE_PL_SET = [set([]) for i in range(2)]
SC=';;'

opts, args = {}, []
cfgr = None
spdr = pm


def load_data(mltl=False, pid=0, spfmt='csr', **kwargs):
	print 'The data is located in %s' % spdr.DATA_PATH
	if (opts.scheme == 'trgs'):
		return load_data_trgs(mltl=mltl, pid=pid, spfmt=spfmt, **kwargs)
	elif (opts.scheme == 'trg'):
		return load_data_trg(mltl=mltl, pid=pid, spfmt=spfmt, **kwargs)
	elif (opts.scheme.startswith('cbow')):
		return load_data_cbow(mltl=mltl, pid=pid, spfmt=spfmt, **kwargs)
		
		
def load_data_trgs(mltl=False, pid=0, spfmt='csr'):
	print 'Loading data...'
	try:
		if (mltl):
			# From combined data file
			train_X, train_Y = spdr.get_data(None, method='trigger', scheme='trgs', from_file=True, dataset='train', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
			dev_X, dev_Y = spdr.get_data(None, method='trigger', scheme='trgs', from_file=True, dataset='dev', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
			test_X = spdr.get_data(None, method='trigger', scheme='trgs', from_file=True, dataset='test', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
		else:
			# From splited data file
			train_Xs, train_Ys = spdr.get_mltl_npz([pid], dataset='train', source=opts.year, task=opts.task, spfmt=spfmt)
			train_X, train_Y = train_Xs[0], train_Ys[0]
			dev_Xs, dev_Ys = spdr.get_mltl_npz([pid], dataset='dev', source=opts.year, task=opts.task, spfmt=spfmt)
			dev_X, dev_Y = dev_Xs[0], dev_Ys[0]
			test_Xs = spdr.get_mltl_npz([pid], dataset='test', source=opts.year, task=opts.task, spfmt=spfmt)
			test_X = test_Xs[0]
	except Exception as e:
		print e
		print 'Can not find the data files!'
		sys.exit(1)
	return train_X, train_Y, dev_X, dev_Y, test_X


def load_data_trg(mltl=False, pid=0, spfmt='csr'):
	print 'Loading data...'
	try:
		train_word_X, train_word_y, train_edge_X, train_edge_Y = spdr.get_data(None, method='trigger', scheme='trg', from_file=True, dataset='train', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
		dev_word_X, dev_word_y, dev_edge_X, dev_edge_Y = spdr.get_data(None, method='trigger', scheme='trg', from_file=True, dataset='dev', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
		test_word_X, test_rawdata = spdr.get_data(None, method='trigger', scheme='trg', from_file=True, dataset='test', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
	except Exception as e:
		print e
		print 'Can not find the data files!'
		sys.exit(1)
	return train_word_X, train_word_y, train_edge_X, train_edge_Y, dev_word_X, dev_word_y, dev_edge_X, dev_edge_Y, test_word_X, test_rawdata
	
	
def load_data_cbow(mltl=False, pid=0, spfmt='csr', ret_field='event', prefix='cbow'):
	print 'Loading data...'
	try:
		train_Xs, train_Y = spdr.get_data(None, method='cbow', from_file=True, ret_field=ret_field, prefix=prefix, dataset='train', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
		dev_Xs, dev_Y = spdr.get_data(None, method='cbow', from_file=True, ret_field=ret_field, prefix=prefix, dataset='dev', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
		test_Xs, test_rawdata = spdr.get_data(None, method='cbow', from_file=True, ret_field=ret_field, prefix=prefix, dataset='test', source=opts.year, task=opts.task, fmt=opts.fmt, spfmt=opts.spfmt)
		if (not mltl):
			train_Y, dev_Y = train_Y.iloc[:,pid].to_frame(), dev_Y.iloc[:,pid].to_frame()
	except Exception as e:
		print e
		print 'Can not find the data files!'
		sys.exit(1)
	return train_Xs, train_Y, dev_Xs, dev_Y, test_Xs, test_rawdata
	

def build_model(mdl_func, mdl_t, mdl_name, tuned=False, pr=None, mltl=False, **kwargs):
	if (tuned and bool(pr)==False):
		print 'Have not provided parameter writer!'
		return None
	if (mltl):
		return OneVsRestClassifier(mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs)), n_jobs=opts.np)
	else:
		return mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs))
		
		
def gen_keras(input_dim, output_dim, model='vecomnet', **kwargs):
    mdl_map = {'vecomnet':(vecomnet.vecomnet_mdl, 'signedclf'), 'vecentnet':(vecomnet.vecentnet_mdl, 'mlclf'), 'mlmt_vecentnet':(vecomnet.mlmt_vecentnet_mdl, 'mlclf')}
    mdl = mdl_map[model]
    return kerasext.gen_mdl(input_dim, output_dim, mdl[0], mdl[1], backend=opts.dend, verbose=opts.verbose, **kwargs)


# Feature Filtering Models
def gen_featfilt(tuned=False, glb_filtnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	filt_names = []
	for filt_name, filter in [
#		('Var Cut', VarianceThreshold()),
#		('Chi2 Pval on FPR', SelectFpr(chi2, alpha=0.05)),
#		('ANOVA-F Pval on FPR', SelectFpr(f_classif, alpha=0.05)),
#		('Chi2 Top K Perc', SelectPercentile(chi2, percentile=30)),
#		('ANOVA-F Top K Perc', SelectPercentile(f_classif, percentile=30)),
#		('Chi2 Top K', SelectKBest(chi2, k=1000)),
#		('ANOVA-F Top K', SelectKBest(f_classif, k=1000)),
#		('LinearSVC', LinearSVC(loss='squared_hinge', dual=False, **pr('Classifier', 'LinearSVC') if tuned else {})),
#		('Logistic Regression', SelectFromModel(LogisticRegression(dual=False, **pr('Feature Selection', 'Logistic Regression') if tuned else {}))),
#		('Lasso', SelectFromModel(LassoCV(cv=6), threshold=0.16)),
#		('Lasso-LARS', SelectFromModel(LassoLarsCV(cv=6))),
#		('Lasso-LARS-IC', SelectFromModel(LassoLarsIC(criterion='aic'), threshold=0.16)),
#		('Randomized Lasso', SelectFromModel(RandomizedLasso(random_state=0))),
#		('Extra Trees Regressor', SelectFromModel(ExtraTreesRegressor(100))),
		# ('U102-GSS502', ftslct.MSelectKBest(ftslct.gen_ftslct_func(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=100), k=500)),
		# ('GSS502', ftslct.MSelectKBest(ftslct.gss_coef, k=500)),
#		('Combined Model', FeatureUnion([('Var Cut', VarianceThreshold()), ('Chi2 Top K', SelectKBest(chi2, k=1000))])),
		('No Feature Filtering', None)
	]:
		yield filt_name, filter
		filt_names.append(filt_name)
	if (len(glb_filtnames) < len(filt_names)):
		del glb_filtnames[:]
		glb_filtnames.extend(filt_names)
		

def gen_trg_featfilt(tuned=False, glb_filtnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	filt_names = []
	for filt_name, filter in [
		('No Feature Filtering', None)
	]:
		yield filt_name, filter
		filt_names.append(filt_name)
	if (len(glb_filtnames) < len(filt_names)):
		del glb_filtnames[:]
		glb_filtnames.extend(filt_names)
		
		
def gen_edge_featfilt(tuned=False, glb_filtnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	filt_names = []
	for filt_name, filter in [
		('No Feature Filtering', None)
	]:
		yield filt_name, filter
		filt_names.append(filt_name)
	if (len(glb_filtnames) < len(filt_names)):
		del glb_filtnames[:]
		glb_filtnames.extend(filt_names)
		
		
def gen_nn_featfilt(tuned=False, glb_filtnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	filt_names = []
	for filt_name, filter in [
		('No Feature Filtering', None)
	]:
		yield filt_name, filter
		filt_names.append(filt_name)
	if (len(glb_filtnames) < len(filt_names)):
		del glb_filtnames[:]
		glb_filtnames.extend(filt_names)


# Classification Models
def gen_clfs(tuned=False, glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clf_names = []
	for clf_name, clf in [
# 		('RidgeClassifier', RidgeClassifier(tol=1e-2, solver='lsqr')),
#		('Perceptron', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np)),
#		('Passive-Aggressive', PassiveAggressiveClassifier(n_iter=50, n_jobs=1 if opts.mltl else opts.np)),
#		('kNN', KNeighborsClassifier(n_neighbors=100, n_jobs=1 if opts.mltl else opts.np)),
#		('NearestCentroid', NearestCentroid()),
#		('BernoulliNB', BernoulliNB()),
#		('MultinomialNB', MultinomialNB()),
#		('ExtraTrees', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np)),
#		('RandomForest', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
		# ('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))])),
		('RandomForest', Pipeline([('clf', RandomForestClassifier(n_estimators=100, n_jobs=opts.np, random_state=0, class_weight='balanced'))])),
#		('BaggingkNN', BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
#		('BaggingLinearSVC', build_model(BaggingClassifier, 'Classifier', 'Bagging LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), n_jobs=1 if opts.mltl else opts.np, random_state=0)(LinearSVC(), max_samples=0.5, max_features=0.5)),
#		('LinSVM', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False)),
#		('RbfSVM', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl))
	]:
		yield clf_name, clf
		clf_names.append(clf_name)
	if (len(glb_clfnames) < len(clf_names)):
		del glb_clfnames[:]
		glb_clfnames.extend(clf_names)
		

def gen_trg_clfs(tuned=False, glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clf_names = []
	for clf_name, clf in [
		('RandomForest', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
		# ('RbfSVM', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl))
	]:
		yield clf_name, clf
		clf_names.append(clf_name)
	if (len(glb_clfnames) < len(clf_names)):
		del glb_clfnames[:]
		glb_clfnames.extend(clf_names)
		
		
def gen_edge_clfs(tuned=False, glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clf_names = []
	for clf_name, clf in [
		('RandomForest', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
		# ('RbfSVM', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl))
	]:
		yield clf_name, clf
		clf_names.append(clf_name)
	if (len(glb_clfnames) < len(clf_names)):
		del glb_clfnames[:]
		glb_clfnames.extend(clf_names)
		
		
# Neural Network Classification model
def gen_nnclf_models(input_dim, output_dim, epochs=1, batch_size=32, **kwargs):
	def nnclf(tuned=False, glb_filtnames=[], glb_cltnames=[]):
		tuned = tuned or opts.best
		common_cfg = cfgr('evnt_extrc', 'common')
		pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
		clt_names = []
		for clt_name, clt in [
			('VeComNet', gen_keras(input_dim, output_dim, model='vecomnet', w2v_path=kwargs.setdefault('w2v_path', 'wordvec.bin'), epochs=epochs, batch_size=batch_size))
		]:
			yield clt_name, clt
			clt_names.append(clt_name)
		if (other_clts is not None):
			for clt_name, clt in other_clts(tuned, glb_filtnames, glb_clfnames):
				yield clt_name, clt
				clt_names.append(clt_name)
		if (len(glb_cltnames) < len(clt_names)):
			del glb_cltnames[:]
			glb_cltnames.extend(clt_names)
	return nnclf
		

# Benchmark Models
def gen_bm_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	# Feature Filtering Model
	for filt_name, filter in gen_featfilt(tuned, glb_filtnames):
		# Classification Model
		for clf_name, clf in gen_clfs(tuned, glb_clfnames):
			yield filt_name, filter, clf_name, clf
			del clf
		del filter
		

def gen_trg_bm_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	# Feature Filtering Model
	for filt_name, filter in gen_trg_featfilt(tuned, glb_filtnames):
	# Classification Model
		for clf_name, clf in gen_trg_clfs(tuned, glb_clfnames):
			yield filt_name, filter, clf_name, clf
			del clf
		del filter
		
def gen_edge_bm_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	# Feature Filtering Model
	for filt_name, filter in gen_edge_featfilt(tuned, glb_filtnames):
	# Classification Model
		for clf_name, clf in gen_edge_clfs(tuned, glb_clfnames):
			yield filt_name, filter, clf_name, clf
			del clf
		del filter
		
		
def gen_nn_bm_models(input_dim, output_dim, epochs=1, batch_size=32, **kwargs):
	def nnbm(tuned=False, glb_filtnames=[], glb_clfnames=[]):
		# Feature Filtering Model
		for filt_name, filter in gen_nn_featfilt(tuned, glb_filtnames):
		# Classification Model
			for clf_name, clf in gen_nnclf_models(input_dim, output_dim, epochs=epochs, batch_size=batch_size)(tuned, glb_clfnames):
				yield filt_name, filter, clf_name, clf
				del clf
			del filter
	return nnbm
		
	
# Combined Models	
def gen_cb_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
#	filtref_func = ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'))
	for mdl_name, mdl in [
		# ('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('UDT-RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, k=500, fn=100)), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RandomForest', Pipeline([('featfilt', SelectFromModel(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0))), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RbfSVM102-2', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 102-2', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM103-2', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 103-2', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM102-3', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 102-3', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM103-3', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 103-3', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('DF-RbfSVM', Pipeline([('featfilt', ftslct.MSelectOverValue(ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'), os.path.join(spdr.DATA_PATH, 'orig_X.npz')))), ('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		('RbfSVM', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('L1-LinSVC', Pipeline([('clf', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False))])),
		('Perceptron', Pipeline([('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np))])),
		('MNB', Pipeline([('clf', build_model(MultinomialNB, 'Classifier', 'MultinomialNB', tuned=tuned, pr=pr, mltl=opts.mltl))])),
#		('5NN', Pipeline([('clf', build_model(KNeighborsClassifier, 'Classifier', 'kNN', tuned=tuned, pr=pr, mltl=opts.mltl, n_neighbors=5, n_jobs=1 if opts.mltl else opts.np))])),
		('MEM', Pipeline([('clf', build_model(LogisticRegression, 'Classifier', 'Logistic Regression', tuned=tuned, pr=pr, mltl=opts.mltl, dual=False))])),
		# ('LinearSVC with L2 penalty [Ft Filt] & Perceptron [CLF]', Pipeline([('featfilt', SelectFromModel(build_model(LinearSVC, 'Feature Selection', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False, penalty='l2'))), ('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, n_jobs=opts.np))])),
		('ExtraTrees', Pipeline([('clf', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np))])),
		('Random Forest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))]))
	]:
		yield mdl_name, mdl
		
		
def gen_cbnn_models(input_dim, output_dim, epochs=1, batch_size=32, **kwargs):
	def cbnn(tuned=False, glb_filtnames=[], glb_clfnames=[]):
		tuned = tuned or opts.best
		common_cfg = cfgr('evnt_extrc', 'common')
		pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
		kw_args = dict([(arg, kwargs[param]) for param, arg in [('class_weight', 'class_weight'), ('pretrain_vecmdl', 'pretrain_vecmdl'), ('precomp_vec', 'precomp_vec'), ('test_ratio', 'validation_split')] if kwargs.has_key(param)])
		if (opts.scheme == 'cbow_ent'):
			models = [
				('VecEntNet', gen_keras(input_dim, output_dim, model=kwargs.setdefault('prefered_mdl', 'vecentnet'), w2v_path=kwargs.setdefault('w2v_path', None), cw2v_path=kwargs.setdefault('cw2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', **func.update_dict(pr('NeuralNetwork', 'VecEntNet') if tuned else {}, kw_args))),
				# ('MLMT VeEntNet', gen_keras(input_dim, output_dim, model='mlmt_' + kwargs.setdefault('prefered_mdl', 'vecentnet'), w2v_path=kwargs.setdefault('w2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', n_jobs=opts.np, **func.update_dict(kw_args, {'mlmt':True})))
			]
		elif (opts.scheme == 'cbow'):
			models = [
				('VeComNet', gen_keras(input_dim, output_dim, model=kwargs.setdefault('prefered_mdl', 'vecomnet'), w2v_path=kwargs.setdefault('w2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', **func.update_dict(pr('NeuralNetwork', 'VeComNet') if tuned else {}, kw_args))),
			]
		for mdl_name, mdl in models:
			yield mdl_name, mdl
	return cbnn


# DNN Models with parameter range
def gen_nnmdl_params(input_dim, output_dim, epochs=1, batch_size=32, rdtune=False, **kwargs):
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	kw_args = dict([(arg, kwargs[param]) for param, arg in [('class_weight', 'class_weight'), ('pretrain_vecmdl', 'pretrain_vecmdl'), ('precomp_vec', 'precomp_vec'), ('test_ratio', 'validation_split')] if kwargs.has_key(param)])
	if (rdtune):
		if (opts.scheme == 'cbow_ent'):
			models_params = [
				('VecEntNet', gen_keras(input_dim, output_dim, model=kwargs.setdefault('prefered_mdl', 'vecentnet'), w2v_path=kwargs.setdefault('w2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', **kw_args), {
					'param_dist':dict(
						lstm_dim=np.logspace(5, 9, num=5, base=2, dtype='int'),
						mlp_dim=np.logspace(5, 9, num=5, base=2, dtype='int'),
						drop_ratio=np.logspace(-0.301, 0, num=10).tolist()),
					'n_iter':32
				})
			]
		elif (opts.scheme == 'cbow'):
			models_params = [
				('VeComNet', gen_keras(input_dim, output_dim, model=kwargs.setdefault('prefered_mdl', 'vecomnet'), w2v_path=kwargs.setdefault('w2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', **kw_args), {
					'param_dist':dict(
						evnt_mlp_dim=np.logspace(5, 9, num=5, base=2, dtype='int'),
						drop_ratio=np.logspace(-0.301, 0, num=10).tolist()),
					'n_iter':32
				})
			]
		for mdl_name, mdl, params in models_params:
			yield mdl_name, mdl, params
	else:
		if (opts.scheme == 'cbow_ent'):
			models_params = [
				('VecEntNet', gen_keras(input_dim, output_dim, model=kwargs.setdefault('prefered_mdl', 'vecentnet'), w2v_path=kwargs.setdefault('w2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', **kw_args), {
					'param_grid':dict(
						lstm_dim=np.logspace(5, 9, num=5, base=2, dtype='int'),
						mlp_dim=np.logspace(5, 9, num=5, base=2, dtype='int'),
						drop_ratio=np.logspace(-0.301, 0, num=10).tolist())
				})
			]
		elif (opts.scheme == 'cbow'):
			models_params = [
				('VeComNet', gen_keras(input_dim, output_dim, model=kwargs.setdefault('prefered_mdl', 'vecomnet'), w2v_path=kwargs.setdefault('w2v_path', None), epochs=epochs, batch_size=batch_size, shuffle='batch', **kw_args), {
					'param_grid':dict(
						evnt_mlp_dim=np.logspace(5, 9, num=5, base=2, dtype='int'),
						drop_ratio=np.logspace(-0.301, 0, num=10).tolist())
				})
			]
		for mdl_name, mdl, params in models_params:
			yield mdl_name, mdl, params
			

# Models with parameter range
def gen_mdl_params(rdtune=False):
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	if (rdtune):
		for mdl_name, mdl, params in [
			# ('Logistic Regression', LogisticRegression(dual=False), {
				# 'param_dist':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('LinearSVC', LinearSVC(dual=False), {
				# 'param_dist':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('Perceptron', Perceptron(), {
				# 'param_dist':dict(
					# alpha=np.logspace(-6, 3, 10),
					# n_iter=stats.randint(3, 20)),
				# 'n_iter':30
			# }),
			# ('MultinomialNB', MultinomialNB(), {
				# 'param_dist':dict(
					# alpha=np.logspace(-6, 3, 10),
					# fit_prior=[True, False]),
				# 'n_iter':30
			# }),
			# ('SVM', SVC(), {
				# 'param_dist':dict(
					# kernel=['linear', 'rbf', 'poly'],
					# C=np.logspace(-5, 5, 11),
					# gamma=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('Extra Trees', ExtraTreesClassifier(random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# class_weight=['balanced', None]),
				# 'n_iter':30
			# }),
			('Random Forest', RandomForestClassifier(random_state=0), {
				'param_dist':dict(
					n_estimators=[50, 100] + range(200, 1001, 200),
					max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					max_depth=[None] + range(10,101,10),
					min_samples_leaf=[1]+range(10, 101, 10),
					class_weight=['balanced', None]),
				'n_iter':30
			}),
			# ('Bagging LinearSVC', BaggingClassifier(base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=opts.best, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_samples=np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6),
					# bootstrap=[True, False],
					# bootstrap_features=[True, False]),
				# 'n_iter':30
			# }),
			# ('AdaBoost LinearSVC', AdaBoostClassifier(base_estimator=build_model(SVC, 'Classifier', 'SVM', tuned=opts.best, pr=pr, mltl=opts.mltl), algorithm='SAMME', random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# learning_rate=np.linspace(0.5, 1, 6)),
				# 'n_iter':30
			# }),
			# ('GB LinearSVC', GradientBoostingClassifier(random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# subsample = np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# learning_rate=np.linspace(0.5, 1, 6),
					# loss=['deviance', 'exponential']),
				# 'n_iter':30
			# }),
			# ('UGSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_dist':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int')),
				# 'n_iter':8
			# }),
		]:
			yield mdl_name, mdl, params
	else:
		for mdl_name, mdl, params in [
			# ('Logistic Regression', LogisticRegression(dual=False), {
				# 'param_grid':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10))
			# }),
			# ('LinearSVC', LinearSVC(dual=False), {
				# 'param_grid':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10))
			# }),
			# ('Perceptron', Perceptron(), {
				# 'param_grid':dict(
					# alpha =np.logspace(-5, 5, 11),
					# n_iter=range(3, 20, 3))
			# }),
			# ('MultinomialNB', MultinomialNB(), {
				# 'param_grid':dict(
					# alpha=np.logspace(-6, 3, 10),
					# fit_prior=[True, False])
			# }),
			# ('SVM', SVC(), {
				# 'param_grid':dict(
					# kernel=['linear', 'rbf', 'poly'],
					# C=np.logspace(-5, 5, 11),
					# gamma=np.logspace(-6, 3, 10))
			# }),
			# ('Extra Trees', ExtraTreesClassifier(random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# class_weight=['balanced', None])
			# }),
			('Random Forest', RandomForestClassifier(random_state=0), {
				'param_grid':dict(
					n_estimators=[50, 100] + range(200, 1001, 200),
					max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					max_depth=[None] + range(10,101,10),
					min_samples_leaf=[1]+range(10, 101, 10),
					class_weight=['balanced', None])
			}),
			# ('Bagging LinearSVC', BaggingClassifier(base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=opts.best, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_samples=np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6),
					# bootstrap=[True, False],
					# bootstrap_features=[True, False])
			# }),
			# ('AdaBoost LinearSVC', AdaBoostClassifier(base_estimator=build_model(SVC, 'Classifier', 'SVM', tuned=opts.best, pr=pr, mltl=opts.mltl), algorithm='SAMME', random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# learning_rate=np.linspace(0.5, 1, 6))
			# }),
			# ('GB LinearSVC', GradientBoostingClassifier(random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# subsample = np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# learning_rate = np.linspace(0.5, 1, 6),
					# loss=['deviance', 'exponential'])
			# }),
			# ('UDT & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('DT & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.decision_tree)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('UNGL & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.ngl_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('NGL & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.ngl_coef)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('UGSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('GSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.gss_coef)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# })
		]:
			yield mdl_name, mdl, params
			
			
def post_process(data, type):
	if (type == 'trigger'):
		trg_pred, test_rawdata = data['trg_pred'], data['test_rawdata']
		edge_offset, edge_bndrys, edges, trg_crsrgmnts = 0, [], [], []
		for docid, corpus, preprcs, word_offset, dist_mt in zip(test_rawdata['docids'], test_rawdata['corpus'], test_rawdata['preprcs'], test_rawdata['word_offset'], test_rawdata['dist_mts']):
			annots, words, depends, sent_bndry = preprcs
			edge_list, edge_bndry, trg_crsrgmnt = [[] for x in range(3)]
			# Find the triggers in a document
			trg_lb = trg_pred[word_offset:word_offset + len(words['str'])]
			# print spdr.print_tokens(words['str'], trg_lb, words['annot_id'])
			for sb in sent_bndry:
				# Construct edges from each sentence
				# tokens = words['str'][sb[0]:sb[1]]
				trg_idx = np.where(trg_lb[sb[0]:sb[1]] == 1)[0].astype('int8').tolist()
				if (len(trg_idx) == 0):
					edge_bndry.append((edge_offset, edge_offset))
					continue
				elif (len(trg_idx) > 1):
					print 'More than one trigger are found in a sentence: [%s]' % (', '.join([words['str'][x] for x in trg_idx]))
				# Edge starts from trigger
				trigger, crs_rgmnt = sb[0] + trg_idx[0], {}
				if (len(words['annot_id'][trigger]) > 0):
					trg_crsrgmnts.append([words['annot_id'][trigger][0], []])
				else:
					trg_crsrgmnts.append([None, []])
				# Find the annotations that contain the word that are connected with the trigger and mark down the annotation id
				for wid in xrange(sb[0], sb[1]):
					if (wid == trigger):
						continue
					for aid in words['annot_id'][wid]:
						if (crs_rgmnt.has_key(aid)):
							continue
						if (dist_mt[trigger, wid] != np.inf):
							crs_rgmnt[aid] = wid
				for aid, wid in crs_rgmnt.iteritems():
					edge_list.append((word_offset + trigger, word_offset + wid))
					trg_crsrgmnts[-1][1].append(aid)
				edge_bndry.append((edge_offset, len(edge_list)))
				edge_offset += len(edge_list)
				edges.extend(edge_list)
			edge_bndrys.append(edge_bndry)
		return edges, edge_bndrys, trg_crsrgmnts
	elif (type == 'edge'):
		edge_preds, edge_bndrys, trg_crsrgmnts, test_rawdata = data['edge_preds'], data['edge_bndrys'], data['trg_crsrgmnts'], data['test_rawdata']
		pred_evnts = []
		# Construct events for different approach
		for edge_pred in edge_preds:
			sent_offset, events = 0, []
			for docid, corpus, preprcs, word_offset, edge_bndry in zip(test_rawdata['docids'], test_rawdata['corpus'], test_rawdata['preprcs'], test_rawdata['word_offset'], edge_bndrys):
				annots, words, depends, sent_bndry = preprcs
				event_list = []
				# Construct events from each sentence
				for sb, eb in zip(sent_bndry, edge_bndry):
					if (eb[1] - eb[0] <= 0):
						sent_offset += 1
						continue
					# Obtain the edge label
					edge_lbs = edge_pred[eb[0]:eb[1]]
					event_oprnds, trg_crsrgmnt = [], trg_crsrgmnts[sent_offset]
					# Extract all the event arguments
					for idx in xrange(len(trg_crsrgmnt[1])):
						if (edge_lbs[idx].sum() > 0):
							event_oprnds.append(trg_crsrgmnt[1][idx])
					# Trigger with an annotation should be considered as an event argument
					if (trg_crsrgmnt[0] != None):
						for crsrgmnt, edge_lb in zip(event_oprnds, edge_lbs):
							event_list.append(dict(loprnd=annots['id'][trg_crsrgmnt[0]], loprndtp=annots['type'][trg_crsrgmnt[0]], roprnd=annots['id'][crsrgmnt], roprndtp=annots['type'][crsrgmnt], type=edge_lb))
					# Concatenate all the annotation in the edges
					for oprand_pair in itertools.combinations(trg_crsrgmnt[1], 2):
						event_list.append(dict(loprnd=annots['id'][oprand_pair[0]], loprndtp=annots['type'][oprand_pair[0]], roprnd=annots['id'][oprand_pair[1]], roprndtp=annots['type'][oprand_pair[1]], type=0))
					sent_offset += 1
				events.append((docid, event_list))
			pred_evnts.append(events)
		return pred_evnts
		
		
def all_trgs():
	global FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET, cfgr

	if (opts.mltl):
		pid = 'all'
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data
	train_X, train_Y, dev_X, dev_Y, test_X = load_data(opts.mltl, pid, opts.spfmt)
	## Union word data
	union_col = func.remove_duplicate(train_X.columns.tolist() + dev_X.columns.tolist() + test_X.columns.tolist())
	X_mask = pd.DataFrame([], columns=union_col)
	train_X = pd.concat([train_X, dev_X, X_mask], copy=False).fillna(0).astype(train_X.dtypes.iloc[0])
	train_Y = pd.concat([train_Y, dev_Y], copy=False).fillna(0).astype(train_Y.dtypes.iloc[0])
	test_X = pd.concat([test_X, X_mask], copy=False).fillna(0).astype(test_X.dtypes.iloc[0])
	## Model building for trigger
	model_iter = gen_cb_models if opts.comb else gen_bm_models
	model_param = dict(tuned=opts.best, glb_filtnames=FILT_NAMES, glb_clfnames=CLF_NAMES)
	global_param = dict(comb=opts.comb, pl_names=PL_NAMES, pl_set=PL_SET)
	## Model training and prediction
	txtclf.cross_validate(train_X, train_Y, model_iter, model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), split_param={'shuffle':True}, global_param=global_param, lbid=pid)
	preds, scores = txtclf.classification(train_X, train_Y, test_X, model_iter, model_param, cfg_param=cfgr('bionlp.txtclf', 'classification'), global_param=global_param, lbid=pid)
	## Save results
	for pl_name, pred in zip(PL_NAMES, preds):
		store_path = os.path.join(spdr.DATA_PATH, 'pred', pl_name)
		fs.mkdir(store_path)
		test_Y = pd.DataFrame(pred, index=test_X.index, columns=train_Y.columns)
		if (len(pred.shape) > 1 and pred.shape[1] > 1):
			true_idx = np.where(pred.sum(axis=1) > 0)[0].tolist()
		else:
			true_idx = np.where(pred > 0)[0].tolist()
		label_cols = test_Y.columns.tolist()
		events = OrderedDict()
		for idx in true_idx:
			docid, t_sub, t_obj = test_Y.index[idx]
			true_lb = np.where(test_Y.iloc[idx] > 0)[0].tolist()
			for lb in true_lb:
				events.setdefault(docid, []).append(dict(subid=t_sub, subtp=label_cols[lb][1], objid=t_obj, objtp=label_cols[lb][2], type=label_cols[lb][0]))
		for docid, event_list in events.iteritems():
			event_str = '\n'.join(['R%i %s %s:%s %s:%s' % (x, event_list[x]['type'], event_list[x]['subtp'], event_list[x]['subid'], event_list[x]['objtp'], event_list[x]['objid']) for x in range(len(event_list))])
			fs.write_file(event_str, os.path.join(store_path, '%s.a2' % docid))


def all_trg():
	global TRG_FILT_NAMES, TRG_CLF_NAMES, EDGE_FILT_NAMES, EDGE_CLF_NAMES, TRG_PL_NAMES, EDGE_PL_NAMES, TRG_PL_SET, EDGE_PL_SET, cfgr

	if (opts.mltl):
		pid = 'all'
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data
	train_word_X, train_word_Y, train_edge_X, train_edge_Y, dev_word_X, dev_word_Y, dev_edge_X, dev_edge_Y, test_word_X, test_rawdata = load_data()
	## Union word data
	union_word_col = func.remove_duplicate(train_word_X.columns.tolist() + dev_word_X.columns.tolist() + test_word_X.columns.tolist())
	word_X_mask = pd.DataFrame([], columns=union_word_col)
	train_word_X = pd.concat([train_word_X, dev_word_X, word_X_mask], copy=False).fillna(0).astype(train_word_X.dtypes.iloc[0])
	train_word_Y = pd.concat([train_word_Y, dev_word_Y], copy=False).fillna(0).astype(train_word_Y.dtypes.iloc[0])
	test_word_X = pd.concat([test_word_X, word_X_mask], copy=False).fillna(0).astype(train_word_X.dtypes.iloc[0])
	## Model building for trigger
	trg_model_iter = gen_cb_models if opts.comb else gen_trg_bm_models
	trg_model_param = dict(tuned=opts.best, glb_filtnames=TRG_FILT_NAMES, glb_clfnames=TRG_CLF_NAMES)
	trg_global_param = dict(comb=opts.comb, pl_names=TRG_PL_NAMES, pl_set=TRG_PL_SET)
	## Trigger training and prediction
	#txtclf.cross_validate(train_word_X, train_word_Y, trg_model_iter, trg_model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), global_param=trg_global_param, lbid=pid)
	trg_preds, trg_scores = txtclf.classification(train_word_X, train_word_Y, test_word_X, trg_model_iter, trg_model_param, cfg_param=cfgr('bionlp.txtclf', 'classification'), global_param=trg_global_param, lbid=pid)
	pred_events, test_edge_Xs, test_edge_Ys  = [[] for i in range(3)]
	for trg_pred in trg_preds:
		## Generate edges for testing set
		test_edges, edge_bndrys, trg_crsrgmnts = post_process(dict(trg_pred=trg_pred, test_rawdata=test_rawdata), 'trigger')
		word_mt, wm_cols, test_edge_data = train_word_X.values, train_word_X.columns.values, np.array(test_edges)
		edge_mt = np.hstack((word_mt[test_edge_data[:,0].astype(int),:], word_mt[test_edge_data[:,1].astype(int),:]))
		em_cols = ['lf_%s' % col for col in wm_cols] + ['rt_%s' % col for col in wm_cols]
		test_edge_X = pd.DataFrame(edge_mt, columns=em_cols)
		## Union edge data
		union_edge_col = func.remove_duplicate(train_edge_X.columns.tolist() + dev_edge_X.columns.tolist() + test_edge_X.columns.tolist())
		edge_X_mask = pd.DataFrame([], columns=union_edge_col)
		train_edge_X = pd.concat([train_edge_X, dev_edge_X, edge_X_mask], copy=False).fillna(0).astype(train_edge_X.dtypes.iloc[0])
		train_edge_Y = pd.concat([train_edge_Y, dev_edge_Y], copy=False).fillna(0).astype(train_edge_Y.dtypes.iloc[0])
		test_edge_X = pd.concat([test_edge_X, edge_X_mask], copy=False).fillna(0).astype(train_edge_X.dtypes.iloc[0])
		test_edge_Xs.append(test_edge_X)
		## Model building for edge
		edge_model_iter = gen_cb_models if opts.comb else gen_edge_bm_models
		edge_model_param = dict(tuned=opts.best, glb_filtnames=EDGE_FILT_NAMES, glb_clfnames=EDGE_CLF_NAMES)
		edge_global_param = dict(comb=opts.comb, pl_names=EDGE_PL_NAMES, pl_set=EDGE_PL_SET)
		## Edge training and prediction
		edge_preds, edge_scores = txtclf.classification(train_edge_X, train_edge_Y, test_edge_X, edge_model_iter, edge_model_param, cfg_param=cfgr('bionlp.txtclf', 'classification'), global_param=edge_global_param, lbid=pid)
		## Generate events
		pred_event = post_process(dict(edge_preds=edge_preds, edge_bndrys=edge_bndrys, trg_crsrgmnts=trg_crsrgmnts, test_rawdata=test_rawdata), 'edge')
		pred_events.append(pred_event)
		test_edge_Y = [pd.DataFrame(edge_pred, columns=train_edge_Y.columns) for edge_pred in edge_preds]
		test_edge_Ys.append(test_edge_Y)
	## Save results
	for trg_pipeline, trg_pred, pred_event, test_edge_X, test_edge_Y_list in zip(TRG_PL_NAMES, trg_preds, pred_events, test_edge_Xs, test_edge_Ys):
		trg_path = os.path.join(spdr.DATA_PATH, 'pred', trg_pipeline)
		fs.mkdir(trg_path)
		trg_pred_df = pd.DataFrame(trg_pred, columns=train_word_Y.columns)
		if (opts.fmt == 'npz'):
			io.write_df(trg_pred_df, os.path.join(trg_path, 'testwY.npz'), sparse_fmt=opts.spfmt)
			io.write_df(test_edge_X, os.path.join(trg_path, 'testeX.npz'), sparse_fmt=opts.spfmt, compress=True)
		else:
			trg_pred_df.to_csv(os.path.join(trg_path, 'testwY.csv'), encoding='utf8')
			test_edge_X.to_csv(os.path.join(trg_path, 'testeX.csv'), encoding='utf8')
		for edge_pl_name, pred_list, test_edge_Y in zip(EDGE_PL_NAMES, pred_event, test_edge_Y_list):
			store_path = os.path.join(spdr.DATA_PATH, 'pred', trg_pipeline, edge_pl_name)
			fs.mkdir(store_path)
			if (opts.fmt == 'npz'):
				io.write_df(test_edge_Y, os.path.join(store_path, 'testeY.npz'), sparse_fmt=opts.spfmt)
			else:
				test_edge_Y.to_csv(os.path.join(store_path, 'testeY.npz'), encoding='utf8')
			for docid, events in pred_list:
				event_str = '\n'.join(['R%i %s %s:%s %s:%s' % (x, train_edge_Y.columns[events[x]['type']], events[x]['loprndtp'], events[x]['loprnd'], events[x]['roprndtp'], events[x]['roprnd']) for x in range(len(events))])
				fs.write_file(event_str, os.path.join(store_path, '%s.a2' % docid))


def all_cbow(prefix='cbow', fusion=False):
	global FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET, cfgr
	
	if (opts.mltl):
		pid = 'all'
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid

	## Load data
	train_Xs, train_Y, dev_Xs, dev_Y, test_Xs, test_rawdata = load_data(mltl=True, ret_field='event', prefix=prefix)
	all_train_Xs, all_train_Y = ([pd.concat([train_X, dev_X]) for train_X, dev_X in zip(train_Xs, dev_Xs)], pd.concat([train_Y, dev_Y]).fillna(0).astype('int8')) if fusion else (train_Xs, train_Y)
	if (not opts.mltl):
		all_train_Y = all_train_Y.iloc[:,pid].to_frame()
		common_cfg = cfgr('evnt_extrc', 'common')
		npg_ratio = common_cfg.setdefault('npg_ratio', None)
		if (npg_ratio is not None):
			if (1.0 * all_train_Y.iloc[:,0].sum() / all_train_Y.shape[0] < 1.0 / (npg_ratio + 1)):
				# all_true = all_train_Y.index[all_train_Y.iloc[:,0] > 0].tolist()
				# all_false = all_train_Y.index[all_train_Y.iloc[:,0] <= 0].tolist()
				all_true = np.arange(all_train_Y.shape[0])[all_train_Y.iloc[:,0] > 0].tolist()
				all_false = np.arange(all_train_Y.shape[0])[all_train_Y.iloc[:,0] <= 0].tolist()
				# Make a cut
				# false_id = np.random.choice(len(all_false), size=int(1.0 * npg_ratio * len(all_true)), replace=False)
				# false_idx = [all_false[i] for i in false_id]
				# all_train_idx = all_true + false_idx
				# Make a compensation
				true_id = np.random.choice(len(all_true), size=int(1.0 / npg_ratio * len(all_false)), replace=True)
				true_idx = [all_true[i] for i in true_id]
				all_train_idx = true_idx + all_false
				all_train_Xs = [x.iloc[all_train_idx] for x in all_train_Xs]
				all_train_Y = all_train_Y.iloc[all_train_idx]
	kwargs, signed = {'prefered_mdl':'vecomnet'}, True
	if (opts.cache is not None and ':' in opts.cache):
		cache_type, cache_path_str = opts.cache.split(':')
		cache_paths = cache_path_str.split(SC)
		if (cache_type == 'model'):
			print 'Using cached model...'
			cache_path = cache_paths[0]
			mdl_clf = io.read_obj(cache_path)
			custom_objects = {}
			custom_objects = func.update_dict(func.update_dict(custom_objects, kerasext.CUSTOM_METRIC), vecomnet.CUSTOM_LOSS)
			mdl_clf.load(os.path.splitext(cache_path)[0], custom_objects=custom_objects)
			kwargs['pretrain_vecmdl'] = mdl_clf.predict_model
		elif (cache_type == 'pred' or cache_type == 'prob'):
			print 'Using cached prediction...'
			data_key = {'pred':'pred_lb', 'prob':'pred_prob'}
			data_path = cache_paths[0]
			preds = [pd.read_hdf(data_path, key=cpath) for cpath in cache_paths[1:]]
			# preds = [io.read_npz(cpath)[data_key[cache_type]] for cpath in cache_paths]
			ds_shapes = [pred.shape[0] for pred in preds]
			ds_shape, ds_count = np.unique(ds_shapes, return_counts=True)
			if (ds_shape.size > 1):
				if (np.unique(ds_count).size > 1):
					raise ValueError('Inconsistant input data sets!')
					print [pred.shape for pred in preds]
				all_train_Xs = kwargs['precomp_vec'] = [pd.concat([preds[j] for j in range(i, len(preds), ds_count[0])]).fillna(0).astype('int8') for i in range(ds_count[0])]
			else:
				all_train_Xs = kwargs['precomp_vec'] = preds
			if (not opts.mltl and 'all_train_idx' in locals()):
				all_train_Xs = [x.loc[all_train_idx] for x in all_train_Xs]
			test_Xs = [pd.read_hdf(data_path, key='cbow/test_pseudo_X%i' % i) for i in range(2)]
		elif (cache_type == 'embed'):
			print 'Using cached embedding...'
			if (pid == 'all' or pid == -1):
				print 'Multi-class/Multi-label classification on argument embedding is not supported yet!'
				sys.exit(-1)
			data_prefix = cache_paths[0].split('|')
			data_path, cntx_prefix = data_prefix[0], data_prefix[1] if len(data_prefix) > 1 else prefix
			all_evnt_args = pd.read_hdf(data_path, key='%s/train_ent_Y' % prefix).columns.tolist()
			evnt_type, lent_type, rent_type = all_train_Y.columns[0].split(':')
			# evnt_args = spdr.EVNT_ARG_TYPE[opts.year][evnt_type]
			evnt_args = [lent_type, rent_type]
			evnt_arg_idx = [all_evnt_args.index(x) for x in evnt_args]
			# Two possible direction / argument order: X0&ArgM+X0&ArgN, X1&ArgN+X1&ArgM
			evnt_arg_idcs = [evnt_arg_idx, evnt_arg_idx[::-1]] # for [X0, X1]
			train_argvecs = [pd.concat([pd.read_hdf(data_path, key='%s/train_argvec%i_X%i' % (cntx_prefix, arg_idx, i)) for arg_idx in arg_idcs], axis=1) for i, arg_idcs in enumerate(evnt_arg_idcs)]
			dev_argvecs = [pd.concat([pd.read_hdf(data_path, key='%s/dev_argvec%i_X%i' % (cntx_prefix, arg_idx, i)) for arg_idx in arg_idcs], axis=1) for i, arg_idcs in enumerate(evnt_arg_idcs)]
			# X0&ArgM+X1&ArgM, X0&ArgN+X1&ArgN
			# train_argvecs = [pd.concat([pd.read_hdf(data_path, key='%s/train_argvec%i_X%i' % (cntx_prefix, arg_idx, i)) for i in range(2)], axis=1) for arg_idx in evnt_arg_idx]
			# dev_argvecs = [pd.concat([pd.read_hdf(data_path, key='%s/dev_argvec%i_X%i' % (cntx_prefix, arg_idx, i)) for i in range(2)], axis=1) for arg_idx in evnt_arg_idx]
			if (fusion):
				all_train_Xs = kwargs['precomp_vec'] = [pd.concat([train_X, dev_X]) for train_X, dev_X in zip(train_argvecs, dev_argvecs)]
				test_Xs = [pd.concat([pd.read_hdf(data_path, key='cbow/test_argvec%i_X%i' % (arg_idx, i)) for arg_idx in arg_idcs], axis=1) for i, arg_idcs in enumerate(evnt_arg_idcs)]
			else:
				all_train_Xs = kwargs['precomp_vec'] = train_argvecs
				test_Xs, test_Y = dev_argvecs, dev_Y
			if (not opts.mltl and 'all_train_idx' in locals()):
				all_train_Xs = [x.iloc[all_train_idx] for x in all_train_Xs]
	print 'Training dataset size of X and Y: %s' % str(([x.shape for x in all_train_Xs], all_train_Y.shape))
	print 'Testing dataset size of X and Y: %s' % str(([x.shape for x in test_Xs], test_Y.shape)) if ('test_Xs' in locals() and 'test_Y' in locals()) else ''

	## Model building
	kwargs.update(dict(input_dim=all_train_Xs[0].shape[1], output_dim=all_train_Y.shape[1] if len(all_train_Y.shape) > 1 else 1, epochs=opts.epoch, batch_size=opts.bsize, evnt_mlp_dim=32, class_weight=np.array([imath.mlb_clsw(all_train_Y.iloc[:,i], norm=True) for i in range(all_train_Y.shape[1])]).reshape((-1,)) if len(all_train_Y.shape)>1 and all_train_Y.shape[1]>1 else imath.mlb_clsw(all_train_Y, norm=True)))
	# kwargs.update(dict(input_dim=all_train_Xs[0].shape[1], output_dim=all_train_Y.shape[1] if len(all_train_Y.shape) > 1 else 1, test_ratio=0.1, epochs=opts.epoch, batch_size=opts.bsize, evnt_mlp_dim=32, class_weight=np.array([imath.mlb_clsw(all_train_Y.iloc[:,i], norm=True) for i in range(all_train_Y.shape[1])]).reshape((-1,)) if len(all_train_Y.shape)>1 and all_train_Y.shape[1]>1 else imath.mlb_clsw(all_train_Y, norm=True)))
	if (opts.cncptw2v is not None and os.path.exists(opts.cncptw2v)): kwargs['cw2v_path'] = opts.cncptw2v
	# Convert the directed labels into binary (optional, only for vecomnet)
	all_train_Y = pd.DataFrame(np.column_stack([np.abs(all_train_Y.values).reshape((all_train_Y.shape[0],-1))] + [label_binarize(lb, classes=[-1,1,0])[:,1] for lb in (np.sign(all_train_Y.values).astype('int8').reshape((all_train_Y.shape[0],-1))).T]), index=all_train_Y.index, columns=all_train_Y.columns.tolist() + ['%s_Dir' % col for col in all_train_Y.columns])
	print 'Modified training dataset size of X and Y: %s' % str(([x.shape for x in all_train_Xs], all_train_Y.shape))
	orig_signed, orig_mltl = signed, opts.mltl
	signed, opts.mltl = False, True
	if (opts.dend is not None):
		print 'DNN model parameters: %s' % {k:v for k, v in kwargs.items() if k != 'pretrain_vecmdl' and k != 'precomp_vec'}
		model_iter = gen_cbnn_models(**kwargs) if opts.comb else gen_nn_bm_models(**kwargs)
	else:
		all_train_Xs = pd.concat(all_train_Xs, axis=1)
		model_iter = gen_cb_models if opts.comb else gen_bm_models
	# all_train_Xs, all_train_Y = [x.iloc[np.random.choice(len(x), size=10000, replace=True)] for x in all_train_Xs], all_train_Y.iloc[np.random.choice(len(all_train_Y), size=10000, replace=True)]
	model_param = dict(tuned=opts.best, glb_filtnames=FILT_NAMES, glb_clfnames=CLF_NAMES)
	global_param = dict(signed=signed, comb=opts.comb, mdl_save_kwargs={'sep_arch':opts.crsdev}, pl_names=PL_NAMES, pl_set=PL_SET)
	print 'Extra model parameters: %s %s' % (model_param, global_param)
	
	## Training and prediction
	if (opts.pred):
		preds, scores = txtclf.classification(all_train_Xs, all_train_Y, test_Xs, model_iter, model_param=model_param, cfg_param=cfgr('bionlp.txtclf', 'classification'), global_param=global_param, lbid='' if orig_mltl else opts.pid)
	else:
		if (opts.eval and 'test_Y' in locals()):
			txtclf.evaluate(all_train_Xs, all_train_Y, test_Xs, test_Y, model_iter, model_param=model_param, cfg_param=cfgr('bionlp.txtclf', 'classification'), global_param=global_param, lbid='' if orig_mltl else opts.pid)
		else:
			txtclf.cross_validate(all_train_Xs, all_train_Y, model_iter, model_param=model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), split_param={'shuffle':True}, global_param=global_param, lbid='' if orig_mltl else opts.pid)
		
	signed, opts.mltl = orig_signed, orig_mltl


def entity_cbow(prefix='cbow', fusion=False):
	global FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET, cfgr
	
	if (opts.mltl):
		pid = 'all'
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid

	## Load data
	train_Xs, train_Y, dev_Xs, dev_Y, test_Xs, test_rawdata = load_data(mltl=True, ret_field='entity', prefix=prefix) # Develop dataset only has parts of the entity labels
	all_train_Xs, all_train_Y = ([pd.concat([train_X, dev_X]) for train_X, dev_X in zip(train_Xs, dev_Xs)], pd.concat([train_Y, dev_Y]).fillna(0).astype('int8')) if fusion else (train_Xs, train_Y)
	if (not opts.mltl):
		all_train_Y = all_train_Y.iloc[:,pid].to_frame()
		common_cfg = cfgr('evnt_extrc', 'common')
		npg_ratio = common_cfg.setdefault('npg_ratio', None)
		if (npg_ratio is not None):
			if (1.0 * all_train_Y.iloc[:,0].sum() / all_train_Y.shape[0] < 1.0 / (npg_ratio + 1)):
				# all_true = all_train_Y.index[all_train_Y.iloc[:,0] > 0].tolist()
				# all_false = all_train_Y.index[all_train_Y.iloc[:,0] <= 0].tolist()
				all_true = np.arange(all_train_Y.shape[0])[all_train_Y.iloc[:,0] > 0].tolist()
				all_false = np.arange(all_train_Y.shape[0])[all_train_Y.iloc[:,0] <= 0].tolist()
				# Make a cut
				# false_id = np.random.choice(len(all_false), size=int(1.0 * npg_ratio * len(all_true)), replace=False)
				# false_idx = [all_false[i] for i in false_id]
				# all_train_idx = all_true + false_idx
				# Make a compensation
				true_id = np.random.choice(len(all_true), size=int(1.0 / npg_ratio * len(all_false)), replace=True)
				true_idx = [all_true[i] for i in true_id]
				all_train_idx = true_idx + all_false
				all_train_Xs = [x.iloc[all_train_idx] for x in all_train_Xs]
				all_train_Y = all_train_Y.iloc[all_train_idx]
	print 'Training dataset size of X and Y: %s' % str(([x.shape for x in all_train_Xs] if type(all_train_Xs) is list else all_train_Xs.shape, all_train_Y.shape))

	## Model building
	kwargs = dict(input_dim=all_train_Xs[0].shape[1], output_dim=all_train_Y.shape[1] if len(all_train_Y.shape) > 1 else 1, epochs=opts.epoch, batch_size=opts.bsize, prefered_mdl='vecentnet', class_weight=np.array([imath.mlb_clsw(all_train_Y.iloc[:,i], norm=True) for i in range(all_train_Y.shape[1])]).reshape((-1,)) if len(all_train_Y.shape)>1 and all_train_Y.shape[1]>1 else imath.mlb_clsw(all_train_Y, norm=True))
	# kwargs = dict(input_dim=all_train_Xs[0].shape[1], output_dim=all_train_Y.shape[1] if len(all_train_Y.shape) > 1 else 1, test_ratio=0.1, epochs=opts.epoch, batch_size=opts.bsize, class_weight=np.array([imath.mlb_clsw(all_train_Y.iloc[:,i], norm=True) for i in range(all_train_Y.shape[1])]).reshape((-1,)) if len(all_train_Y.shape)>1 and all_train_Y.shape[1]>1 else imath.mlb_clsw(all_train_Y, norm=True))
	if (opts.cncptw2v is not None and os.path.exists(opts.cncptw2v)): kwargs['cw2v_path'] = opts.cncptw2v
	if (opts.dend is not None):
		print 'DNN model parameters: %s' % kwargs
		model_iter = gen_cbnn_models(**kwargs) if opts.comb else gen_nn_bm_models(**kwargs)
	else:
		all_train_Xs = pd.concat(all_train_Xs, axis=1)
		model_iter = gen_cb_models if opts.comb else gen_bm_models
	# all_train_Xs, all_train_Y = [x.iloc[np.random.choice(len(x), size=10000, replace=True)] for x in all_train_Xs], all_train_Y.iloc[np.random.choice(len(all_train_Y), size=10000, replace=True)]
	model_param = dict(tuned=opts.best, glb_filtnames=FILT_NAMES, glb_clfnames=CLF_NAMES)
	global_param = dict(comb=opts.comb, mdl_save_kwargs={'sep_arch':opts.crsdev}, pl_names=PL_NAMES, pl_set=PL_SET)
	
	## Training and prediction
	if (opts.pred):
		preds, scores = txtclf.classification(all_train_Xs, all_train_Y, test_Xs, model_iter, model_param=model_param, cfg_param=cfgr('bionlp.txtclf', 'classification'), global_param=global_param, lbid='' if opts.mltl else opts.pid)
	else:
		txtclf.cross_validate(all_train_Xs, all_train_Y, model_iter, model_param=model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), split_param={'shuffle':True}, global_param=global_param, lbid='' if opts.mltl else opts.pid)


def all_entry():
	if (opts.scheme == 'trg'):
		all_trg()
	elif (opts.scheme == 'trgs'):
		all_trgs()
	elif (opts.scheme == 'cbow'):
		all_cbow(fusion=opts.fusion)
	elif (opts.scheme == 'cbow_ent'):
		entity_cbow(fusion=opts.fusion)
		
		
def tuning():
	if (opts.scheme == 'trg'):
		tuning_trg()
	elif (opts.scheme == 'trgs'):
		tuning_trgs()
	elif (opts.scheme == 'cbow'):
		tuning_cbow()
	elif (opts.scheme == 'cbow_ent'):
		tuning_cbowent()
		
		
def tuning_trg(fusion=False):
	pass


def tuning_trgs(fusion=False):
	pass


def tuning_cbow(fusion=False):
	pass


def tuning_cbowent(fusion=False):
	from sklearn.model_selection import KFold
	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data for entity
	train_Xs, train_Y, dev_Xs, dev_Y, test_Xs, test_rawdata = load_data(mltl=True, ret_field='entity') # Develop dataset only has parts of the entity labels
	all_train_Xs, all_train_Y = ([pd.concat([train_X, dev_X]) for train_X, dev_X in zip(train_Xs, dev_Xs)], pd.concat([train_Y, dev_Y]).fillna(0).astype('int8')) if fusion else (train_Xs, train_Y)
	if (not opts.mltl):
		all_train_Y = all_train_Y.iloc[:,pid].to_frame()
		common_cfg = cfgr('evnt_extrc', 'common')
		npg_ratio = common_cfg.setdefault('npg_ratio', None)
		if (npg_ratio is not None):
			if (1.0 * all_train_Y.iloc[:,0].sum() / all_train_Y.shape[0] < 1.0 / (npg_ratio + 1)):
				all_true = np.arange(all_train_Y.shape[0])[all_train_Y.iloc[:,0] > 0].tolist()
				all_false = np.arange(all_train_Y.shape[0])[all_train_Y.iloc[:,0] <= 0].tolist()
				true_id = np.random.choice(len(all_true), size=int(1.0 / npg_ratio * len(all_false)), replace=True)
				true_idx = [all_true[i] for i in true_id]
				all_train_idx = true_idx + all_false
				all_train_Xs = [x.iloc[all_train_idx] for x in all_train_Xs]
				all_train_Y = all_train_Y.iloc[all_train_idx]
	print 'Training dataset size of X and Y: %s' % str(([x.shape for x in all_train_Xs] if type(all_train_Xs) is list else all_train_Xs.shape, all_train_Y.shape))
	
	## Parameter tuning for entity
	print 'Parameter tuning for entity is starting ...'
	kwargs = dict(input_dim=all_train_Xs[0].shape[1], output_dim=all_train_Y.shape[1] if len(all_train_Y.shape) > 1 else 1, epochs=opts.epoch, batch_size=opts.bsize, prefered_mdl='vecentnet', class_weight=np.array([imath.mlb_clsw(all_train_Y.iloc[:,i], norm=True) for i in range(all_train_Y.shape[1])]).reshape((-1,)) if len(all_train_Y.shape)>1 and all_train_Y.shape[1]>1 else imath.mlb_clsw(all_train_Y, norm=True))
	ext_params = dict(cv=KFold(n_splits=opts.kfold, shuffle=True, random_state=0))
	params_generator = gen_mdl_params() if opts.dend is None else gen_nnmdl_params(**kwargs)
	for mdl_name, mdl, params in params_generator:
		params.update(ext_params)
		print 'Tuning hyperparameters for %s' % mdl_name
		pt_result = txtclf.tune_param(mdl_name, mdl, all_train_Xs, all_train_Y, opts.rdtune, params, mltl=opts.mltl, avg=opts.avg, n_jobs=opts.np)
		io.write_npz(dict(zip(['best_params', 'best_score', 'score_avg_cube', 'score_std_cube', 'dim_names', 'dim_vals'], pt_result)), 'cbowent_%s_param_tuning_for_%s_%s' % (opts.solver.lower().replace(' ', '_'), mdl_name.replace(' ', '_').lower(), 'all' if (pid == -1) else pid))
	
	
def regen(data_path, ent_Xs, ent_Y, evnt_Xs, evnt_Y, ent_split_key, evnt_split_key, n_splits=3):
	import evnt_helper as helper
	prefix = 'cbow_regen_%if' % n_splits
	for i in range(evnt_Y.shape[1]):
		evnt_y = evnt_Y.iloc[:,i].to_frame()
		lbid_str = '' if (len(evnt_Y.shape) == 1 or evnt_Y.shape[1] == 1) else '/%i' % i
		for j, ([train_ents, train_evnts], [test_ents, test_evnts]) in enumerate(helper._split_ents_evnts(ent_Xs[0].index.tolist(), evnt_Xs[0].index.tolist(), ent_split_key, evnt_split_key, n_splits)):
			train_ent_Xs, test_ent_Xs = [x.loc[train_ents] for x in ent_Xs], [x.loc[test_ents] for x in ent_Xs]
			train_evnt_Xs, test_evnt_Xs = [x.loc[train_evnts] for x in evnt_Xs], [x.loc[test_evnts] for x in evnt_Xs]
			train_ent_y, test_ent_y = ent_Y.loc[train_ents], ent_Y.loc[test_ents]
			train_evnt_y, test_evnt_y = evnt_y.loc[train_evnts], evnt_y.loc[test_evnts]
			_ = [df.to_hdf(data_path, '%s/%i%s/%s_ent_X%i' % (prefix, j, lbid_str, 'train', k), format='table', data_columns=True) for k, df in enumerate(train_ent_Xs)]
			_ = [df.to_hdf(data_path, '%s/%i%s/%s_X%i' % (prefix, j, lbid_str, 'train', k), format='table', data_columns=True) for k, df in enumerate(train_evnt_Xs)]
			_ = [df.to_hdf(data_path, '%s/%i%s/%s_ent_X%i' % (prefix, j, lbid_str, 'dev', k), format='table', data_columns=True) for k, df in enumerate(test_ent_Xs)]
			_ = [df.to_hdf(data_path, '%s/%i%s/%s_X%i' % (prefix, j, lbid_str, 'dev', k), format='table', data_columns=True) for k, df in enumerate(test_evnt_Xs)]
			_ = [df.to_hdf(data_path, '%s/%i%s/%s_ent_X%i' % (prefix, j, lbid_str, 'test', k), format='table', data_columns=True) for k, df in enumerate(test_ent_Xs)]
			_ = [df.to_hdf(data_path, '%s/%i%s/%s_X%i' % (prefix, j, lbid_str, 'test', k), format='table', data_columns=True) for k, df in enumerate(test_evnt_Xs)]
			train_evnt_y.to_hdf(data_path, '%s/%i%s/%s_Y' % (prefix, j, lbid_str, 'train'), format='table', data_columns=True)
			train_ent_y.to_hdf(data_path, '%s/%i%s/%s_ent_Y' % (prefix, j, lbid_str, 'train'), format='table', data_columns=True)
			test_evnt_y.to_hdf(data_path, '%s/%i%s/%s_Y' % (prefix, j, lbid_str, 'dev'), format='table', data_columns=True)
			test_ent_y.to_hdf(data_path, '%s/%i%s/%s_ent_Y' % (prefix, j, lbid_str, 'dev'), format='table', data_columns=True)
	
	
def demo():
	def _clear_globals():
		global FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET
		FILT_NAMES, CLF_NAMES, PL_NAMES = [[] for x in range(3)]
		PL_SET = set([])
		
	import evnt_helper as helper
	## Download the pre-generated datasets
	
	## Read meta data
	data_path = os.path.join(spdr.DATA_PATH, opts.year, opts.task, 'dataset.h5')
	all_evnt_args = pd.read_hdf(data_path, key='cbow/train_ent_Y').columns.tolist()
	all_events = pd.read_hdf(data_path, key='cbow/train_Y').columns.tolist()
	prefix = cntx_prefix = 'cbow_regen' if opts.regen else 'cbow'
	to_pred = opts.pred
	# Cross-validation
	# opts.pred = False
	# opts.scheme = 'cbow_ent'
	# for i in range(len(all_evnt_args)):
		# opts.pid = i
		# entity_cbow(fusion=True)
		# _clear_globals()
	
	# opts.scheme = 'cbow'
	# for i in range(len(all_events)):
		# opts.pid = i
		# all_cbow(fusion=True)
		# _clear_globals()
	
	## Prediction on development dataset
	opts.pred = True
	opts.scheme = 'cbow_ent'
	if (opts.regen): # Regenerate train and dev dataset to cbow_regen
		train_ent_Xs, train_ent_Y, dev_ent_Xs, dev_ent_Y, test_ent_Xs, test_ent_rawdata = load_data(mltl=True, ret_field='entity', prefix='cbow')
		all_train_ent_Xs, all_train_ent_Y = ([pd.concat([train_X, dev_X]) for train_X, dev_X in zip(train_ent_Xs, dev_ent_Xs)], pd.concat([train_ent_Y, dev_ent_Y]).fillna(0).astype('int8')) if opts.fusion else (train_ent_Xs, train_ent_Y)
		train_evnt_Xs, train_evnt_Y, dev_evnt_Xs, dev_evnt_Y, test_evnt_Xs, test_evnt_rawdata = load_data(mltl=True, ret_field='event', prefix='cbow')
		all_train_evnt_Xs, all_train_evnt_Y = ([pd.concat([train_X, dev_X]) for train_X, dev_X in zip(train_evnt_Xs, dev_evnt_Xs)], pd.concat([train_evnt_Y, dev_evnt_Y]).fillna(0).astype('int8')) if opts.fusion else (train_evnt_Xs, train_evnt_Y)
		split_key = lambda x: '-'.join(x.split('|')[0].split('-')[:2])
		regen(data_path, all_train_ent_Xs, all_train_ent_Y, all_train_evnt_Xs, all_train_evnt_Y, split_key, split_key, n_splits=opts.kfold)
		all_train_evnt_stat = all_train_evnt_Y.abs().sum(axis=0)
		train_ent_evnt_map = [[(i, evnt) for i, evnt in enumerate(all_events) if ent in evnt] for ent in all_evnt_args]
	for i in range(len(all_evnt_args)):
		if (opts.cache is not None and opts.cache == 'skip'): break
		opts.pid = i
		for j in range(opts.kfold):
			if (opts.regen): # Regenerate the train and dev dataset in different folder to train argument embedding
				invlv_evnt_ids, invlv_evnt = zip(*train_ent_evnt_map[i])
				invlv_evnt = list(invlv_evnt) if len(invlv_evnt) > 1 else invlv_evnt[0]
				cntx_prefix = 'cbow_regen_%if/%i' % (opts.kfold, j) 
				prefix = cntx_prefix if (len(all_train_evnt_Y.shape) == 1 or all_train_evnt_Y.shape[1] == 1) else '%s/%i' % (cntx_prefix, all_train_evnt_stat.index.get_loc(all_train_evnt_stat.loc[invlv_evnt].argmax()) if len(invlv_evnt_ids) > 1 else invlv_evnt_ids[0])
				#prefix = 'cbow_regen_%if/%i' % (opts.kfold, j) if (len(all_train_evnt_Y.shape) == 1 or all_train_evnt_Y.shape[1] == 1) else 'cbow_regen_%if/%i/%i' % (opts.kfold, j, all_train_evnt_stat.index.get_loc(all_train_evnt_stat.loc[invlv_evnt].argmax()) if len(invlv_evnt_ids) > 1 else invlv_evnt_ids[0])
			if not (opts.cache is not None and os.path.exists(opts.cache) and os.path.exists(os.path.join(opts.cache, 'clf_pred_vecentnet_%i.npz' % i))):
				_clear_globals()
				entity_cbow(prefix=prefix, fusion=opts.fusion and to_pred)
				mdl_dir = '.'
			else:
				mdl_dir = opts.cache
			with kerasext.gen_cntxt(opts.dend, **dict(device='/cpu:0')): # Store the argument embedding in the unified location
				helper._contex2vec(os.path.join(mdl_dir, 'vecentnet_clf_%i.pkl' % i), os.path.join(spdr.DATA_PATH, opts.year, opts.task, 'dataset.h5'), ['%s/train_X%i' % (prefix, k) for k in range(4)], cntxvec_fpath='%s/train_' % cntx_prefix, crsdev=opts.crsdev)
				helper._contex2vec(os.path.join(mdl_dir, 'vecentnet_clf_%i.pkl' % i), os.path.join(spdr.DATA_PATH, opts.year, opts.task, 'dataset.h5'), ['%s/dev_X%i' % (prefix, k) for k in range(4)], cntxvec_fpath='%s/dev_' % cntx_prefix, crsdev=opts.crsdev)
				if (opts.fusion):
					helper._contex2vec(os.path.join(mdl_dir, 'vecentnet_clf_%i.pkl' % i), os.path.join(spdr.DATA_PATH, opts.year, opts.task, 'dataset.h5'), ['%s/test_X%i' % (prefix, k) for k in range(4)], cntxvec_fpath='%s/test_' % cntx_prefix, crsdev=opts.crsdev)

	## Event prediction
	# opts.pred = to_pred
	# opts.scheme = 'cbow'
	# orig_cache = opts.
	# prefix = cntx_prefix = 'cbow_regen' if opts.regen else 'cbow'
	# opts.cache = 'embed:%s|%s' % (data_path, cntx_prefix)
	##prefix = 'cbow' if not opts.regen else ('cbow_regen' if (len(all_events) == 1) else 'cbow_regen_0')
	# orig_wd = os.getcwd()
	# for i in range(len(all_events)):
		# new_wd = os.path.join(orig_wd, str(i))
		# fs.mkdir(new_wd)
		# os.chdir(new_wd)
		# opts.pid = 0 if opts.regen else i
		# sub_orig_wd = os.getcwd()
		# for j in range(opts.kfold):
			# new_wd = os.path.join(sub_orig_wd, str(j))
			# fs.mkdir(new_wd)
			# os.chdir(new_wd)
			# _clear_globals()
			# opts.cache = 'embed:%s|%s' % (data_path, cntx_prefix)
			# if (opts.regen):
				# cntx_prefix = 'cbow_regen_%if/%i' % (opts.kfold, j) 
				# prefix = cntx_prefix if (len(all_train_evnt_Y.shape) == 1 or all_train_evnt_Y.shape[1] == 1) else '%s/%i' % (cntx_prefix, all_train_evnt_stat.index.get_loc(all_train_evnt_stat.loc[invlv_evnt].argmax()) if len(invlv_evnt_ids) > 1 else invlv_evnt_ids[0])
			# all_cbow(prefix=prefix, fusion=opts.fusion and to_pred) # no need to fuse if regenerate train and dev
			# os.chdir(sub_orig_wd)
		# os.chdir(orig_wd)
	# opts.cache = orig_cache
	# if (to_pred):
		# fnames = ['clf_pred_vecomnet_%i.npz' % i for i in range(len(all_events))]
		# preds = [io.read_npz(fname)['pred_lb'] for fname in fnames]
		# probs = [io.read_npz(fname)['pred_prob'] for fname in fnames]
		# pred = np.column_stack(preds)
		# prob = np.column_stack(probs)
		# pred_fpath = 'combined_pred_%s' % fnames[0].split('pred_')[1].strip('_0')
		# io.write_npz(dict(pred_lb=pred, pred_prob=prob), pred_fpath)
		# helper._pred2event(spdr, True, pred_fpath, data_path, test_X_paths=['cbow/test_X%i' % i for i in range(4)] if opts.fusion else ['%s/dev_X%i' % (prefix, i) for i in range(4)], train_Y_path='cbow/train_Y', method='cbow', source=opts.year, task=opts.task)
	

def main():
	if (opts.tune):
		tuning()
		return
	if (opts.method == 'demo'):
		demo()
		return
	all_entry()


if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-k', '--kfold', default=10, action='store', type='int', dest='kfold', help='indicate the K fold cross validation')
	op.add_option('-p', '--pid', default=0, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv, npz, or h5 [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
	op.add_option('-t', '--tune', action='store_true', dest='tune', default=False, help='firstly tune the hyperparameters')
	op.add_option('-r', '--rdtune', action='store_true', dest='rdtune', default=False, help='randomly tune the hyperparameters')
	op.add_option('--solver', default='particle_swarm', action='store', type='str', dest='solver', help='solver used to tune the hyperparameters: particle_swarm, grid_search, or random_search, etc.')
	op.add_option('-b', '--best', action='store_true', dest='best', default=False, help='use the tuned hyperparameters')
	op.add_option('-c', '--comb', action='store_true', dest='comb', default=False, help='run the combined methods')
	op.add_option('-l', '--mltl', action='store_true', dest='mltl', default=False, help='use multilabel strategy')
	op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
	op.add_option('-e', '--scheme', default='cbow', type='str', dest='scheme', help='the scheme to generate data')
	op.add_option('-d', '--dend', dest='dend', help='deep learning backend: tf or th')
	op.add_option('-j', '--epoch', default=1, action='store', type='int', dest='epoch', help='indicate the epoch used in deep learning')
	op.add_option('-z', '--bsize', default=32, action='store', type='int', dest='bsize', help='indicate the batch size used in deep learning')
	op.add_option('-o', '--omp', action='store_true', dest='omp', default=False, help='use openmp multi-thread')
	op.add_option('-g', '--gpunum', default=0, action='store', type='int', dest='gpunum', help='indicate the gpu device number')
	op.add_option('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
	op.add_option('--gpumem', default=0.4826445576329565, action='store', type='float', dest='gpumem', help='indicate the per process gpu memory fraction')
	op.add_option('--crsdev', action='store_true', dest='crsdev', default=False, help='whether to use heterogeneous devices')
	op.add_option('-w', '--cache', type='str', dest='cache', help='the pretrained model path or partial prediction path')
	op.add_option('--cncptw2v', dest='cncptw2v', help='indicate whether use the concept embedding separately')
	op.add_option('-i', '--input', default='bnlpst', help='input source: bnlpst or pbmd [default: %default]')
	op.add_option('-y', '--year', default='2016', help='the year when the data is released: 2016 or 2011 [default: %default]')
	op.add_option('-u', '--task', default='bb', help='the year when the data is released: 2016 or 2011 [default: %default]')
	op.add_option('-x', '--pred', action='store_true', dest='pred', default=False, help='train the model and make predictions without cross-validation')
	op.add_option('--eval', action='store_true', dest='eval', default=False, help='evaluation the model on the prepared training and development/testing dataset')
	op.add_option('--fusion', action='store_true', dest='fusion', default=False, help='combine the training set with the development set for training process')
	op.add_option('--regen', action='store_true', dest='regen', default=False, help='regenerate the training and development set')
	op.add_option('-m', '--method', help='main method to run')
	op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')

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
		if (len(w2v_cfg) > 0):
			if (w2v_cfg['DATA_PATH'] is not None and w2v_cfg['W2V_MODEL'] is not None and os.path.exists(os.path.join(w2v_cfg['DATA_PATH'], w2v_cfg['W2V_MODEL']))):
				w2v.DATA_PATH = w2v_cfg['DATA_PATH']
				w2v.W2V_MODEL = w2v_cfg['W2V_MODEL']
			if (w2v_cfg['MAX_CONN'] is not None):
				w2v.MAX_CONN = w2v_cfg['MAX_CONN']	
			if (w2v_cfg['MAX_TRIAL'] is not None):
				w2v.MAX_TRIAL = w2v_cfg['MAX_TRIAL']
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		txtclf.init(plot_cfg=plot_cfg, plot_common=plot_common)

	if (opts.dend is not None):
		if (opts.dend == 'th' and opts.gpunum == 0 and opts.omp):
			from multiprocessing import cpu_count
			os.environ['OMP_NUM_THREADS'] = '4' if opts.tune else str(int(1.5 * cpu_count() / opts.np))
		if (opts.gpuq is not None and not opts.gpuq.strip().isspace()):
			gpuq = [int(x) for x in opts.gpuq.split(',') if x]
			# dev_id = gpuq[opts.pid % len(gpuq)]
			dev_id = range(len(gpuq))[opts.pid % len(gpuq)]
		else:
			dev_id = opts.pid % opts.gpunum if opts.gpunum > 0 else 0
		kerasext.init(dev_id=dev_id, backend=opts.dend, num_gpu=opts.gpunum, gpuq=gpuq if opts.gpuq is not None else [0], gpu_mem=opts.gpumem, num_process=opts.np, use_omp=opts.omp, verbose=opts.verbose)

	main()