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

from bionlp import txtclf
from bionlp import ftslct
from bionlp.util import fs
from bionlp.util import io
from bionlp.util import func
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


def load_data(mltl=False, pid=0, spfmt='csr'):
	if (opts.scheme == 'trgs'):
		return load_data_trgs(mltl=mltl, pid=pid, spfmt=spfmt)
	elif (opts.scheme == 'trg'):
		return load_data_trg(mltl=mltl, pid=pid, spfmt=spfmt)
		
		
def load_data_trgs(mltl=False, pid=0, spfmt='csr'):
	print 'Loading data...'
	try:
		if (mltl):
			# From combined data file
			train_X, train_Y = spdr.get_data(None, from_file=True, dataset='train', fmt=opts.fmt, spfmt=opts.spfmt)
			dev_X, dev_Y = spdr.get_data(None, from_file=True, dataset='dev', fmt=opts.fmt, spfmt=opts.spfmt)
			test_X = spdr.get_data(None, from_file=True, dataset='test', fmt=opts.fmt, spfmt=opts.spfmt)
		else:
			# From splited data file
			train_Xs, train_Ys = spdr.get_mltl_npz([pid], dataset='train', spfmt=spfmt)
			train_X, train_Y = train_Xs[0], train_Ys[0]
			dev_Xs, dev_Ys = spdr.get_mltl_npz([pid], dataset='dev', spfmt=spfmt)
			dev_X, dev_Y = dev_Xs[0], dev_Ys[0]
			test_Xs = spdr.get_mltl_npz([pid], dataset='test', spfmt=spfmt)
			test_X = test_Xs[0]
	except Exception as e:
		print e
		print 'Can not find the data files!'
		exit(1)
	return train_X, train_Y, dev_X, dev_Y, test_X


def load_data_trg(mltl=False, pid=0, spfmt='csr'):
	print 'Loading data...'
	try:
		train_word_X, train_word_y, train_edge_X, train_edge_Y = spdr.get_data(None, from_file=True, dataset='train', fmt=opts.fmt, spfmt=opts.spfmt)
		dev_word_X, dev_word_y, dev_edge_X, dev_edge_Y = spdr.get_data(None, from_file=True, dataset='dev', fmt=opts.fmt, spfmt=opts.spfmt)
		test_word_X, test_rawdata = spdr.get_data(None, from_file=True, dataset='test', fmt=opts.fmt, spfmt=opts.spfmt)
	except Exception as e:
		print e
		print 'Can not find the data files!'
		exit(1)
	return train_word_X, train_word_y, train_edge_X, train_edge_Y, dev_word_X, dev_word_y, dev_edge_X, dev_edge_Y, test_word_X, test_rawdata
	
	
def build_model(mdl_func, mdl_t, mdl_name, tuned=False, pr=None, mltl=False, **kwargs):
	if (tuned and bool(pr)==False):
		print 'Have not provided parameter writer!'
		return None
	if (mltl):
		return OneVsRestClassifier(mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs)), n_jobs=opts.np)
	else:
		return mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs))


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


# Classification Models
def gen_clfs(tuned=False, glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clf_names = []
	for clf_name, clf in [
#		('RidgeClassifier', RidgeClassifier(tol=1e-2, solver='lsqr')),
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
		
	
# Combined Models	
def gen_cb_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('evnt_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
#	filtref_func = ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'))
	for mdl_name, mdl in [
		# ('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		('UDT-RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, k=500, fn=100)), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RandomForest', Pipeline([('featfilt', SelectFromModel(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0))), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RbfSVM102-2', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 102-2', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM103-2', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 103-2', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM102-3', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 102-3', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM103-3', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 103-3', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('DF-RbfSVM', Pipeline([('featfilt', ftslct.MSelectOverValue(ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'), os.path.join(spdr.DATA_PATH, 'orig_X.npz')))), ('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		('RbfSVM', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('L1-LinSVC', Pipeline([('clf', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False))])),
		# ('Perceptron', Pipeline([('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np))])),
		# ('MNB', Pipeline([('clf', build_model(MultinomialNB, 'Classifier', 'MultinomialNB', tuned=tuned, pr=pr, mltl=opts.mltl))])),
#		('5NN', Pipeline([('clf', build_model(KNeighborsClassifier, 'Classifier', 'kNN', tuned=tuned, pr=pr, mltl=opts.mltl, n_neighbors=5, n_jobs=1 if opts.mltl else opts.np))])),
		# ('MEM', Pipeline([('clf', build_model(LogisticRegression, 'Classifier', 'Logistic Regression', tuned=tuned, pr=pr, mltl=opts.mltl, dual=False))])),
		# ('LinearSVC with L2 penalty [Ft Filt] & Perceptron [CLF]', Pipeline([('featfilt', SelectFromModel(build_model(LinearSVC, 'Feature Selection', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False, penalty='l2'))), ('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, n_jobs=opts.np))])),
		# ('ExtraTrees', Pipeline([('clf', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np))])),
#		('Random Forest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))]))
	]:
		yield mdl_name, mdl


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
		
		
def all():
	if (opts.scheme == 'trgs'):
		all_trgs()
	elif (opts.scheme == 'trg'):
		all_trg()
		
		
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
	exit(0)
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

	
def tuning():
	pass
	
	
def demo():
	pass
	

def main():
	if (opts.tune):
		tuning()
		return
	if (opts.method == 'demo'):
		demo()
		return
	all()


if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-k', '--kfold', default=10, action='store', type='int', dest='kfold', help='indicate the K fold cross validation')
	op.add_option('-p', '--pid', default=0, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
	op.add_option('-t', '--tune', action='store_true', dest='tune', default=False, help='firstly tune the hyperparameters')
	op.add_option('-r', '--rdtune', action='store_true', dest='rdtune', default=False, help='randomly tune the hyperparameters')
	op.add_option('-b', '--best', action='store_true', dest='best', default=False, help='use the tuned hyperparameters')
	op.add_option('-c', '--comb', action='store_true', dest='comb', default=False, help='run the combined methods')
	op.add_option('-l', '--mltl', action='store_true', dest='mltl', default=False, help='use multilabel strategy')
	op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
	op.add_option('-e', '--scheme', default='trgs', type='str', dest='scheme', help='the scheme to generate data')
	op.add_option('-i', '--input', default='bnlpst', help='input source: bnlpst or pbmd [default: %default]')
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
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		txtclf.init(plot_cfg=plot_cfg, plot_common=plot_common)
		
	main()