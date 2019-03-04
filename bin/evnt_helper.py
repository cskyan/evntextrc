#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: evnt_helper.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-11-03 22:47:13
###########################################################################
#

import os
import sys
import ast
import glob
import logging
import itertools
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from bionlp.spider import w2v
from bionlp.util import fs, io, func, plot, shell
import bionlp.spider.pubmed as pm

import bnlpst

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'bnlpst':bnlpst, 'pbmd':pm}
SC=';;'

opts, args = {}, []
cfgr = None
spdr = pm


def init_plot(plot_cfg={}, plot_common={}):
	if (len(plot_cfg) > 0 and plot_cfg['MON'] is not None):
		plot.MON = plot_cfg['MON']
	global plot_common_cfg
	if (len(plot_common) > 0):
		plot_common_cfg = plot_common
		
		
def npzs2yaml(dir_path='.', mdl_t='Classifier'):
	pw = io.param_writer(os.path.join(dir_path, 'mdlcfg'))
	for file in fs.listf(dir_path):
		if file.endswith(".npz"):
			fpath = os.path.join(dir_path, file)
			params = io.read_npz(fpath)['best_params'].tolist()
			for k in params.keys():
				if (type(params[k]) == np.ndarray):
					params[k] == params[k].tolist()
				if (isinstance(params[k], np.generic)):
					params[k] = np.asscalar(params[k])
			pw(mdl_t, file, params)
	pw(None, None, None, True)


def solr(sh='bash'):
	common_cfg = cfgr('evnt_helper', 'common')
	proc_cmd = 'solr stop -all && solr restart && sleep 10s'
	cmd = proc_cmd if sh == 'sh' else '%s -c "%s"' % (sh, proc_cmd)
	shell.daemon(cmd, 'solr.jetty')


def gen_wordvec():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	w2v.pubmed_w2v_solr('http://localhost:8983/solr/pubmed', cache_path=opts.cache, n_jobs=opts.np, **kwargs)
	
	
def _split_ents_evnts(ent_idx, evnt_idx, ent_split_key, evnt_split_key, n_splits=3):
    from sklearn.model_selection import GroupKFold
    evnt_grps = map(evnt_split_key, evnt_idx)
    ent_map = dict([(key, grp.tolist()) for key, grp in pd.Series(range(len(ent_idx)), index=ent_idx).groupby(ent_split_key)])
    group_kfold = GroupKFold(n_splits=n_splits)
    for train_index, test_index in group_kfold.split(X=evnt_idx, groups=evnt_grps):
        train_evnts, test_evnts = [evnt_idx[i] for i in train_index], [evnt_idx[i] for i in test_index]
        train_keys, test_keys = list(set([evnt_grps[i] for i in train_index])), list(set([evnt_grps[i] for i in test_index]))
        train_ents, test_ents = [ent_idx[i] for i in func.flatten_list([ent_map[k] for k in test_keys])], [ent_idx[i] for i in func.flatten_list([ent_map[k] for k in test_keys])]
        yield [train_ents, train_evnts], [test_ents, test_evnts]


def _contex2vec(mdl_path, fpath, X_paths, cntxvec_fpath=None, crsdev=False):
	from keras.layers import Input, concatenate
	from keras.models import Model
	import keras.backend as K
	from bionlp.model import kerasext, vecomnet
	kerasext.init(backend='tf')
	mdl_idx = int(os.path.splitext(mdl_path)[0].split('_')[-1])
	Xs = [pd.read_hdf(fpath, xpath) for xpath in X_paths]
	clf = io.read_obj(mdl_path)

	custom_objects = {}
	custom_objects = func.update_dict(func.update_dict(custom_objects, kerasext.CUSTOM_METRIC), vecomnet.CUSTOM_LOSS)
	clf.load(os.path.splitext(mdl_path)[0], custom_objects=custom_objects, sep_arch=crsdev)
	output = clf.model.get_layer(name='MLP-L1').output
	new_mdl = Model(clf.model.input, output)
	# print new_mdl.summary()
	embeddings = [new_mdl.predict(Xs[i:i+2]) for i in range(0, len(Xs), 2)]
	# print [ebd.shape for ebd in embeddings]
	embed_dfs = [pd.DataFrame(embed, index=Xs[0].index) for embed in embeddings]
	# Store the argument embeddings for each event sample with the format: #DATASET_argvec#ARGGLOBALID_X#ARGLOCALID
	cntxvec_fpath = cntxvec_fpath if cntxvec_fpath else X_paths[0].split('X')[0]
	_ = [df.to_hdf(fpath, '%sargvec%i_X%i' % (cntxvec_fpath, mdl_idx, i), format='table') for i, df in enumerate(embed_dfs)]
	# _ = [io.write_df(df, os.path.join(os.path.dirname(fpath), 'argvec%i_X%i' % (mdl_idx, i)), with_idx=True, compress=True) for i, df in enumerate(embed_dfs)]
	del clf, new_mdl

def contex2vec():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	mdl_path, X_paths = kwargs['mdl'], kwargs['X'].split(SC)
	_contex2vec(mdl_path, opts.loc, X_paths, crsdev=opts.crsdev)


def combine_mdls():
	from keras.layers import Input, concatenate
	from keras.models import Model, clone_model
	import keras.backend as K
	from bionlp.model import kerasext, vecomnet
	kerasext.init(backend='tf')
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	fnames = kwargs['mdls'].split(SC)
	mdl_prefix, out_name = kwargs.setdefault('mdl_prefix', 'Obj'), kwargs.setdefault('out_name', 'Output')
	mdls = [io.read_obj(os.path.join(opts.loc, fname)) for fname in fnames]
	inputs = []
	for i, mdl, fname in zip(range(len(mdls)), mdls, fnames):
		custom_objects = {}
		custom_objects = func.update_dict(func.update_dict(custom_objects, kerasext.CUSTOM_METRIC), vecomnet.CUSTOM_LOSS)
		mdl.load(os.path.join(opts.loc, os.path.splitext(fname)[0]), custom_objects=custom_objects)
		# inputs.append([mdl.model.get_layer(name='X0').output, mdl.model.get_layer(name='X1').output])
		for layer in mdl.model.layers:
			layer.name = '%s%i-%s' % (mdl_prefix, i, layer.name)
		print mdl.model.summary()
	common_input = [Input(shape=K.int_shape(mdls[0].model.input[0])[1:], dtype='int64', name='X%i'%i) for i in range(2)]
	num_in, num_newin = len(mdls[0].model.input), len(common_input)
	# merge_out = [mdl.model(common_input) for i, mdl in enumerate(mdls)]
	# merge_out = []
	cloned_mdls = []
	for i, mdl in enumerate(mdls):
		# Not usable because of keras model still contains the old inputs after I replace it in the clone_model function
		# for x_in in mdl.model.input:
			# mdl.model.layers.pop(0)
		cloned_mdls.append(clone_model(mdl.model, input_tensors=common_input))
		cloned_mdls[-1].name = '%s%i' % (mdl_prefix, i)
		# num_layers = len(cloned_mdls[-1]._nodes_by_depth.keys())
		# del cloned_mdls[-1]._nodes_by_depth[num_layers - 2]
		# cloned_mdls[-1]._nodes_by_depth[num_layers - 2] = cloned_mdls[-1]._nodes_by_depth[num_layers - 1]
		# del cloned_mdls[-1]._nodes_by_depth[num_layers - 1]
		# del cloned_mdls[-1].layers_by_depth[num_layers - 2]
		# cloned_mdls[-1].layers_by_depth[num_layers - 2] = cloned_mdls[-1].layers_by_depth[num_layers - 1]
		# del cloned_mdls[-1].layers_by_depth[num_layers - 1]
		# del cloned_mdls[-1].layers[num_newin:num_newin+num_in]
		# merge_out.append(cloned_mdls[-1](common_input))
	del mdls[:]
	outputs = concatenate([mdl.output for mdl in cloned_mdls], name=out_name)
	comb_mdl = Model(common_input, outputs)
	comb_mdl.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[kerasext.f1])
	comb_mdl.trainable = False
	# comb_mdl.get_layer('%s0-X0'%mdl_prefix).name, comb_mdl.get_layer('%s0-X1'%mdl_prefix).name = 'X0', 'X1'
	comb_clf = kerasext.MLClassifier(build_fn=vecomnet.vecentnet_mdl)
	comb_clf.model = comb_mdl
	print comb_clf.model.summary()
	comb_clf.save('combined_model')
	

def combine_preds():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	fnames = kwargs['preds'].split(SC)
	preds = [io.read_npz(os.path.join(opts.loc, fname))['pred_lb'] for fname in fnames]
	probs = [io.read_npz(os.path.join(opts.loc, fname))['pred_prob'] for fname in fnames]
	pred = np.column_stack(preds)
	prob = np.column_stack(probs)
	io.write_npz(dict(pred_lb=pred, pred_prob=prob), 'combined_pred_%s' % fnames[0].split('pred_')[1].strip('_0'))


def pred2pseudo():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	pred = io.read_npz(kwargs['pred'])[kwargs.setdefault('pred_type', 'pred_prob')]
	if (kwargs['pred_type'] == 'pred_prob' and kwargs.setdefault('exclb', True)):
		multi_class_pred = np.zeros_like(pred)
		multi_class_pred[np.arange(len(pred)), pred.argmax(axis=1)] = 1
		pred = multi_class_pred
	train_pseudo_Xs = [pd.read_hdf(opts.loc, dspath) for dspath in kwargs.setdefault('pseudo_X', ['cbow/train_pseudo_X%i' % i for i in range(2)])]
	test_ent_Xs = [pd.read_hdf(opts.loc, dspath) for dspath in kwargs.setdefault('ent_X', ['cbow/test_ent_X%i' % i for i in range(2)])]
	test_Xs = [pd.read_hdf(opts.loc, dspath) for dspath in kwargs.setdefault('X', ['cbow/test_X%i' % i for i in range(4)])]
	pred_df = pd.DataFrame(pred, index=test_ent_Xs[0].index, columns=train_pseudo_Xs[0].columns)
	lents, rents = [], []
	for idx in test_Xs[0].index:
		docid, lidx, ridx = idx.split('|')
		lents.append('|'.join([docid, lidx]))
		rents.append('|'.join([docid, ridx]))
	test_pseudo_Xs = [pred_df.loc[lents], pred_df.loc[rents]]
	_ = [df.to_hdf(opts.loc, 'cbow/test_pseudo_X%i' % i, format='table', data_columns=True) for i, df in enumerate(test_pseudo_Xs)]
	
	
def _pred2event(spdr_mod, combined, pred_fpath, data_path, test_X_paths=['cbow/dev_X%i' % i for i in range(4)], train_Y_path='cbow/train_Y', method='cbow', source='2011', task='bgi'):
	pred = io.read_npz(pred_fpath)['pred_lb']
	if (combined):
		event_mt = np.column_stack([pred[:,i] for i in range(0, pred.shape[1], 2)])
		dir_mt = np.column_stack([pred[:,i] for i in range(1, pred.shape[1], 2)])
	else:
		evnt_num = pred.shape[1] / 2
		event_mt = pred[:,:evnt_num]
		dir_mt = pred[:,evnt_num:]
	np.place(dir_mt, dir_mt==0, [-1])
	event_mt *= dir_mt
	test_Xs = [pd.read_hdf(data_path, dspath) for dspath in test_X_paths]
	train_Y = pd.read_hdf(data_path, train_Y_path)
	test_Y = pd.DataFrame(event_mt, index=test_Xs[0].index, columns=train_Y.columns)
	io.write_df(test_Y, 'test_Y', with_idx=True, sparse_fmt='csr', compress=True)
	events = spdr_mod.pred2data(test_Y, method=method, source=source, task=task)
	spdr_mod.to_a2(events, './pred', source=source, task=task)
	
	
def pred2event():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	combined = kwargs.setdefault('combined', True)
	_pred2event(spdr, combined, kwargs['pred'], opts.loc, test_X_paths=kwargs.setdefault('X', ['cbow/dev_X%i' % i for i in range(4)]), train_Y_path=kwargs.setdefault('Y', 'cbow/train_Y'), method=opts.scheme, source=opts.year, task=opts.task)

	
def main():
	if (opts.method is None):
		return
	elif (opts.method == 'solr'):
		solr()
	elif (opts.method == 'w2v'):
		gen_wordvec()
	elif (opts.method == 'c2v'):
		contex2vec()
	elif (opts.method == 'combm'):
		combine_mdls()
	elif (opts.method == 'combp'):
		combine_preds()
	elif (opts.method == 'p2p'):
		pred2pseudo()
	elif (opts.method == 'p2e'):
		pred2event()
	

if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-p', '--pid', default=-1, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for calculation')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-c', '--cfg', help='config string used in the plot function, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
	op.add_option('-l', '--loc', default='.', help='the files in which location to be process')
	op.add_option('-o', '--output', default='.', help='the path to store the data')
	op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
	op.add_option('-e', '--scheme', default='cbow', type='str', dest='scheme', help='the scheme to generate data')
	op.add_option('-d', '--dend', dest='dend', help='deep learning backend: tf or th')
	op.add_option('-g', '--gpunum', default=0, action='store', type='int', dest='gpunum', help='indicate the gpu device number')
	op.add_option('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue')
	op.add_option('--gpumem', default=0.4826445576329565, action='store', type='float', dest='gpumem', help='indicate the per process gpu memory fraction')
	op.add_option('--crsdev', action='store_true', dest='crsdev', default=False, help='whether to use heterogeneous devices')
	op.add_option('-i', '--input', default='bnlpst', help='input source: bnlpst or pbmd [default: %default]')
	op.add_option('-y', '--year', default='2016', help='the year when the data is released: 2016 or 2011 [default: %default]')
	op.add_option('-u', '--task', default='bb', help='the year when the data is released: 2016 or 2011 [default: %default]')
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
		w2v.MAX_CONN = w2v_cfg['MAX_CONN']
		w2v.MAX_TRIAL = w2v_cfg['MAX_TRIAL']
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		init_plot(plot_cfg=plot_cfg, plot_common=plot_common)
		
	if (opts.dend is not None):
		if (opts.dend == 'th' and opts.gpunum == 0):
			from multiprocessing import cpu_count
			os.environ['OMP_NUM_THREADS'] = '4' if opts.tune else str(int(1.5 * cpu_count() / opts.np))
		if (opts.gpuq is not None):
			gpuq = [int(x) for x in opts.gpuq.split(',')]
			dev_id = gpuq[opts.pid % len(gpuq)]
		else:
			dev_id = opts.pid % opts.gpunum if opts.gpunum > 0 else 0
		kerasext.init(dev_id=dev_id, num_gpu=opts.gpunum, backend=opts.dend, num_process=opts.np, use_omp=True, verbose=opts.verbose)

	main()