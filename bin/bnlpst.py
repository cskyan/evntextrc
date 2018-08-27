#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2015 by Caspar. All rights reserved.
# File Name: bnlpst.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2015-11-23 21:44:57
###########################################################################
#

import os, re, sys, math, string, logging, operator, itertools
from bisect import bisect
from optparse import OptionParser
from collections import OrderedDict

import numpy as np
import scipy as sp
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, label_binarize

from bionlp.spider import w2v
from bionlp.util import fs, io, func
from bionlp import nlp


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bioevent\\bnlpst'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bioevent', 'bnlpst')
EVNT_LB = {'2016':['Lives_In'], '2013':['Localization', 'PartOf'], '2011':['RegulonDependence', 'BindTo', 'TranscriptionFrom', 'RegulonMember', 'SiteOf', 'TranscriptionBy', 'PromoterOf', 'PromoterDependence', 'ActionTarget', 'Interaction']}
EVNT_ARG_TYPE = {'2016':{'Lives_In':['Bacteria', 'Location']}, '2013':{'Localization':['Bacterium', 'Localization'], 'PartOf':['Host', 'Part']}, '2011':{'RegulonDependence':['Regulon', 'Target'], 'BindTo':['Agent', 'Target'], 'TranscriptionFrom':['Transcription', 'Site'], 'RegulonMember':['Regulon', 'Member'], 'SiteOf':['Site', 'Entity'], 'TranscriptionBy':['Transcription', 'Agent'], 'PromoterOf':['Promoter', 'Gene'], 'PromoterDependence':['Promoter', 'Protein'], 'ActionTarget':['Action', 'Target'], 'Interaction':['Agent', 'Target']}}

[STEM, POS, SPOS, ANNOT_TYPE, ANNOT_HASH] = [set([]) for x in range(5)]
ft_offset = {'train':[], 'dev':[], 'test':[]}
dist_mts = {}
prdcss = {}


# Get the annotations, segmentations, and the dependcies from a1 file
def get_preprcs(docids, dataset='train', source='2016', task='bb', method='spacy', disable=[]):
	preprcs = []
	if (dataset == 'train'):
		dir_path = os.path.join(DATA_PATH, source, task, 'train')
	elif (dataset == 'dev'):
		dir_path = os.path.join(DATA_PATH, source, task, 'dev')
	elif (dataset == 'test'):
		dir_path = os.path.join(DATA_PATH, source, task, 'test')
	for did in docids:
		preprcs.append(get_a1(os.path.join(dir_path, did + '.a1'), source=source, method=method, disable=disable))
	return preprcs


def get_a1(fpath, source='2016', task='bb', method='spacy', disable=[]):
	if (source == '2016'):
		return get_a1_2016(fpath, method=method, disable=disable)
	elif (source == '2013'):
		return get_a1_2013(fpath, method=method, disable=disable)
	elif (source == '2011'):
		return get_a1_2011(fpath, method=method, disable=disable)
	
	
def get_a1_2016(fpath, method='spacy', disable=[]):
	try:
		title, [text, sent_bndry, gddfs, coref] = '', [[] for i in range(4)]
		annots = {'id':[], 'type':[], 'loc':[], 'str':[], 'tkns':[]}	# Biological term annotation, a.k.a. named entity
		words = {'id':[], 'loc':[], 'str':[], 'stem':[], 'pos':[], 'stem_pos':[], 'annot_id':[]} # Word tokenization
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (record[1] == 'Title'):
					title = ' '.join(record[4:])
					continue
				elif (record[1] == 'Paragraph'):
					text.append(' '.join(record[4:]))
					continue
				else:
					annots['id'].append(record[0])
					annots['type'].append(record[1])
					# Extract multiple location pairs of one annotation
					loc_start, loc_list = int(record[2]), []
					for i in xrange(3, len(record)):
						if (';' in record[i]):
							loc_left, loc_right = [int(x) for x in record[i].split(';')][:2]
							loc_list.append((loc_start, loc_left))
							loc_start = loc_right
						else:
							loc_list.append((loc_start, int(record[i])))
							annots['str'].append(' '.join(record[(i + 1):]).strip())
							break
					annots['loc'].append(loc_list)
	except:
		print 'Can not open the file \'%s\'!' % fpath
		sys.exit(-1)
	
	if ('parse' in disable): return sent_bndry, words, annots, gddfs, coref

	content = nlp.clean_text(title + '\n' + '\n'.join(text), encoding='' if method == 'spacy' else 'ascii')
	# Replace the tokens that have character '.' to avoid incorrect sentence separation
	for old_text, new_text in [(' %c.' % x, ' %cx' % x) for x in string.ascii_uppercase] + [(' gen.', ' genx'), (' Lb.', ' Lbx'), (' nov.', ' novx'), (' sp.', ' spx')]:
		content = content.replace(old_text, new_text)
	tokens, gddfs, coref = nlp.parse_all(content, method=method, cached_id=os.path.splitext(os.path.basename(fpath))[0], cache_path=os.path.abspath(os.path.join(fpath, os.path.pardir, '.parsed')), disable=['tag', 'entity'])
	sent_bndry = np.cumsum([0] + [len(sent) for sent in tokens])
	for token_list in tokens:
		# words['str'].extend([token['str'] for token in token_list])
		parsed_words = [content[token['loc'][0]:token['loc'][1]] for token in token_list]
		words_res = nlp.del_punct(parsed_words, ret_idx=True)
		if (len(words_res) > 0): continue
		words_wop, wop_idx = words_res
		words['str'].extend(words_wop)
		words['loc'].extend([token_list[i]['loc'] for i in wop_idx])
		words['stem'].extend([token_list[i]['stem'] for i in wop_idx])
		words['pos'].extend([token_list[i]['pos'] for i in wop_idx])
		words['stem_pos'].extend([token_list[i]['stem_pos'] for i in wop_idx])

	# Map the annotations to the tokens
	annots['tkns'], words['annot_id'] = nlp.annot_align(annots['loc'], words['loc'])

	return sent_bndry, words, annots, gddfs, coref
	
	
def get_a1_2013(fpath, method='spacy', disable=[]):
	try:
		title, [text, sent_bndry, gddfs, coref] = '', [[] for i in range(4)]
		annots = {'id':[], 'type':[], 'loc':[], 'str':[], 'tkns':[]}	# Biological term annotation, a.k.a. named entity
		words = {'id':[], 'loc':[], 'str':[], 'stem':[], 'pos':[], 'stem_pos':[], 'annot_id':[]} # Word tokenization
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				annots['id'].append(record[0])
				annots['type'].append(record[1])
				# Extract multiple location pairs of one annotation
				loc_start, loc_list = int(record[2]), []
				for i in xrange(3, len(record)):
					if (';' in record[i]):
						loc_left, loc_right = [int(x) for x in record[i].split(';')][:2]
						loc_list.append((loc_start, loc_left))
						loc_start = loc_right
					else:
						loc_list.append((loc_start, int(record[i])))
						annots['str'].append(' '.join(record[(i + 1):]).strip())
						break
				annots['loc'].append(loc_list)
	except:
		print 'Can not open the file \'%s\'!' % fpath
		sys.exit(-1)
	
	if ('parse' in disable): return sent_bndry, words, annots, gddfs, coref

	content = nlp.clean_text(''.join(fs.read_file('%s.txt' % os.path.splitext(fpath)[0])), encoding='' if method == 'spacy' else 'ascii')
	# Replace the tokens that have character '.' to avoid incorrect sentence separation
	for old_text, new_text in [(' %c.' % x, ' %cx' % x) for x in string.ascii_uppercase] + [(' gen.', ' genx'), (' Lb.', ' Lbx'), (' nov.', ' novx'), (' sp.', ' spx')]:
		content = content.replace(old_text, new_text)
	tokens, gddfs, coref = nlp.parse_all(content, method=method, cached_id=os.path.splitext(os.path.basename(fpath))[0], cache_path=os.path.abspath(os.path.join(fpath, os.path.pardir, '.parsed')), disable=['tag', 'entity'])
	sent_bndry = np.cumsum([0] + [len(sent) for sent in tokens])
	for token_list in tokens:
		# words['str'].extend([token['str'] for token in token_list])
		parsed_words = [content[token['loc'][0]:token['loc'][1]] for token in token_list]
		words_res = nlp.del_punct(parsed_words, ret_idx=True)
		if (len(words_res) == 0): continue
		words_wop, wop_idx = words_res
		words['str'].extend(words_wop)
		words['loc'].extend([token_list[i]['loc'] for i in wop_idx])
		words['stem'].extend([token_list[i]['stem'] for i in wop_idx])
		words['pos'].extend([token_list[i]['pos'] for i in wop_idx])
		words['stem_pos'].extend([token_list[i]['stem_pos'] for i in wop_idx])

	# Map the annotations to the tokens
	annots['tkns'], words['annot_id'] = nlp.annot_align(annots['loc'], words['loc'], error=1)

	return sent_bndry, words, annots, gddfs, coref
	

def get_a1_2011(fpath, method='spacy', disable=[]):
	try:
		sent_bndry, gddfs, coref = [[] for i in range(3)]
		annots = {'id':[], 'type':[], 'loc':[], 'str':[], 'tkns':[]}	# Biological term annotation, a.k.a. named entity
		words = {'id':[], 'loc':[], 'str':[], 'stem':[], 'pos':[], 'stem_pos':[], 'annot_id':[]} # Word tokenization
		depends = {'id':[], 'type':[], 'oprnd':[]} # Syntactic dependencies
		dpnd_mt = {}
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (line[0] == 'T'):
					annots['id'].append(record[0])
					annots['type'].append(record[1])
					annots['loc'].append([(int(record[2]), int(record[3]))])
					annots['str'].append(' '.join(record[4:]).strip())
				if (line[0] == 'W'):
					words['id'].append(record[0])
					words['loc'].append((int(record[2]), int(record[3])))
					words['str'].append(' '.join(record[4:]))
					words['pos'].append('')
				if (line[0] == 'R'):
					depends['id'].append(record[0])
					depends['type'].append(record[1])
					depends['oprnd'].append((record[2], record[3]))
					nlp.set_mt_point(words['id'].index(record[3]), words['id'].index(record[2]), record[1], dpnd_mt)
	except:
		print 'Can not open the file \'%s\'!' % fpath
		sys.exit(-1)
	sent_bndry = np.array([0, len(words['str'])])
	# Map the annotations to the tokens
	annots['tkns'], words['annot_id'] = nlp.annot_align(annots['loc'], words['loc'])
	
	if ('parse' in disable): return sent_bndry, words, annots, gddfs, coref

	# Generate the Stems of tokens
	if ('stem' not in disable):
		words['stem'] = nlp.stem(words['str'])
		words['stem_pos'] = [sp[1] for sp in nlp.pos(words['stem'])]
	# Extract the part-of-speech from the annotations for every word
	if ('pos' not in disable):
		oprnds = []
		pos_list = []
		for i in xrange(len(depends['id'])):
			operands, dpnd_tp = depends['oprnd'][i], depends['type'][i]
			pos = []
			#%%% Bad RE design, to be modified %%%#
			match_rs = re.compile('.+:(.+)-(.+)\((.*)\)').match(dpnd_tp)
			if (match_rs == None):
				match_rs = re.compile('.+:(.+)-(.+)').match(dpnd_tp)
			if (match_rs == None):
				# Appositive, expressed as 'AS_$synonym'
				match_rs = re.compile('(appos)').match(dpnd_tp)
				words['pos'][words['id'].index(operands[0])] = 'AS_%s' % operands[1]
				words['pos'][words['id'].index(operands[1])] = 'AS_%s' % operands[0]
				continue
			else:
				pos.extend(match_rs.groups()[:2])
			#%%%%%%#
			if (len(pos) != len(operands)):
				print "Annotation Error!"
				continue
			oprnds.extend(operands)
			pos_list.extend(pos)
		for x in xrange(len(oprnds)):
			words['pos'][words['id'].index(oprnds[x])] = pos_list[x]

		# Deal with appositive, link the synonym together into a list, assign the pos to each synonym according to the one has non-appos pos 
		for x in xrange(len(words['pos'])):
			match_rs = re.compile('^AS_(.*)').match(words['pos'][x])
			identcls = [x]
			while (match_rs):
				wid = words['id'].index(match_rs.groups()[0])
				# found the beginning word again
				if (len(identcls) > 1 and identcls[0] == wid):
					break
				identcls.append(wid)
				match_rs = re.compile('^AS_(.*)').match(words['pos'][identcls[-1]])
			if (not match_rs): # The last identical word has non-appos pos
				for y in identcls:
					words['pos'][y] = words['pos'][identcls[-1]]
			else:
				for y in identcls:	# The last identical word does not have non-appos pos, namely found a cycle link
					words['pos'][y] = ''
				continue
			
	# Grammatical dependency matrix
	if ('dependency' not in disable):
		gdmt = nlp.dpnd_trnsfm(dpnd_mt, (len(words['str']), len(words['str'])))
		gddfs = [pd.DataFrame(gdmt.tocsr().todense())]
	return sent_bndry, words, annots, gddfs, coref


# Get the events from a2 file	
def get_evnts(docids, dataset='train', source='2016', task='bb'):
	event_list = []
	if (dataset == 'train'):
		dir_path = os.path.join(DATA_PATH, source, task, 'train')
	elif (dataset == 'dev'):
		dir_path = os.path.join(DATA_PATH, source, task, 'dev')
	elif (dataset == 'test'):
		dir_path = os.path.join(DATA_PATH, source, task, 'test')
	for did in docids:
		event_list.append(get_a2(os.path.join(dir_path, did + '.a2'), source=source))
	return event_list

	
def get_a2(fpath, source='2016', task='bb'):
	if (source == '2016'):
		return get_a2_2016(fpath)
	elif (source == '2013'):
		return get_a2_2013(fpath)
	elif (source == '2011'):
		return get_a2_2011(fpath)
	
	
def get_a2_2016(fpath):
	try:
		events = {'id':[], 'type':[], 'oprnd_tps':[], 'oprnds':[], 'oprnd_words':[]} # Interaction events
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (line[0] == 'R'):
					events['id'].append(record[0])
					events['type'].append(record[1])
					loprnd, roprnd = record[2].split(':'), record[3].split(':')
					events['oprnd_tps'].append((loprnd[0], roprnd[0]))
					events['oprnds'].append((loprnd[1], roprnd[1]))
					events['oprnd_words'].append([[], []])
	except:
		print 'Can not open the file \'%s\'!' % fpath
		sys.exit(-1)
	return events
	
	
def get_a2_2013(fpath):
	try:
		events = {'id':[], 'type':[], 'oprnd_tps':[], 'oprnds':[], 'oprnd_words':[]} # Interaction events
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (line[0] == 'R'):
					events['id'].append(record[0])
					events['type'].append(record[1])
					loprnd, roprnd = record[2].split(':'), record[3].split(':')
					events['oprnd_tps'].append((loprnd[0], roprnd[0]))
					events['oprnds'].append((loprnd[1], roprnd[1]))
					events['oprnd_words'].append([[], []])
	except:
		print 'Can not open the file \'%s\'!' % fpath
		sys.exit(-1)
	return events
	

def get_a2_2011(fpath):
	try:
		events = {'id':[], 'type':[], 'oprnd_tps':[], 'oprnds':[], 'oprnd_words':[]} # Interaction events
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (line[0] == 'E'):
					events['id'].append(record[0])
					events['type'].append(record[1])
					loprnd, roprnd = record[2].split(':'), record[3].split(':')
					events['oprnd_tps'].append((loprnd[0], roprnd[0]))
					events['oprnds'].append((loprnd[1], roprnd[1]))
					events['oprnd_words'].append([[], []])
	except:
		print 'Can not open the file \'%s\'!' % fpath
		sys.exit(-1)
	return events
	

# Get the document ids
def get_docid(dataset='train', source='2016', task='bb'):
	if (dataset == 'train'):
		files = [os.path.splitext(fpath)[0] for fpath in os.listdir(os.path.join(DATA_PATH, source, task, 'train')) if re.match(r'.*\.txt', fpath)]
	elif (dataset == 'dev'):
		files = [os.path.splitext(fpath)[0] for fpath in os.listdir(os.path.join(DATA_PATH, source, task, 'dev')) if re.match(r'.*\.txt', fpath)]
	elif (dataset == 'test'):
		files = [os.path.splitext(fpath)[0] for fpath in os.listdir(os.path.join(DATA_PATH, source, task, 'test')) if re.match(r'.*\.txt', fpath)]
	return files
	

# Get the document text
def get_corpus(docids, dataset='train', source='2016', task='bb', ext_fmt='txt'):
	corpus = []
	if (dataset == 'train'):
		dir_path = os.path.join(DATA_PATH, source, task, 'train')
	elif (dataset == 'dev'):
		dir_path = os.path.join(DATA_PATH, source, task, 'dev')
	elif (dataset == 'test'):
		dir_path = os.path.join(DATA_PATH, source, task, 'test')
	for did in docids:
		corpus.append(' '.join(fs.read_file(os.path.join(dir_path, did+'.%s'%ext_fmt), 'utf8')))
	return corpus
	
	
def get_data(raw_data, method='cbow', scheme='', **kwargs):
	if (method == 'trigger'):
		if (scheme == 'trgs'):
			return get_data_trgs(raw_data, **kwargs)
		elif (scheme == 'trg'):
			return get_data_trg(raw_data, **kwargs)
	elif (method == 'cbow'):
		return get_data_cbow(raw_data, **kwargs)


## Start Trigger-based Approach ##
POS_WEIGHT = {'universal':{'VERB':1, 'ADP':1}, 'stanford':{'VBP':1, 'IN':1}, 'bnlpst2011':{'V\(\w+\)|V_PASS|V':1}}
NOUN_POS = {'universal':'NOUN', 'stanford':'NN|NNS', 'bnlpst2011':'N\(\w+\)|V'}
	
# Get the trigger from a specific file
def get_trgwd(fpath):
	trg_wds = []
	try:
		with open(fpath, 'r') as fd:
			trg_wds.extend(fd.readline().split())
	except:
		print 'Can not open the file \'%s\'!' % fpath
	return trg_wds


def _find_trg(words, wid1, wid2, dist_mt, prdcs, source='2016', task='bb', pos_type='universal'):
	global POS_WEIGHT
	r_pos = re.compile(r'\b^'+r'$\b|\b^'.join(POS_WEIGHT[pos_type].keys())+r'$\b', flags=re.I | re.X)
	trg_wds = get_trgwd(os.path.join(os.path.join(DATA_PATH, source, task, 'trigger'), 'TRIGGERWORDS.txt'))
	r_patn = re.compile(r'\b^'+r'$\b|\b^'.join(trg_wds)+r'$\b', flags=re.I | re.X)
	trg_evntoprt = {'2016':['None'],'2011':['Action']}
	r_evntoprt = re.compile(r'(?=('+'|'.join(trg_evntoprt[source])+r'))', flags=re.I | re.X)
	trigger, prdc = None, wid2
	if (r_pos.match(words['pos'][wid1]) or r_evntoprt.findall(','.join(words['evnt_annotp'][wid1])) or r_patn.findall(words['str'][wid1]) or r_patn.findall(words['stem'][wid1]) or r_pos.match(words['stem_pos'][wid1])):
		return wid1
	elif (r_pos.match(words['pos'][wid2]) or r_evntoprt.findall(','.join(words['evnt_annotp'][wid2])) or r_patn.findall(words['str'][wid2]) or r_patn.findall(words['stem'][wid2]) or r_pos.match(words['stem_pos'][wid2])):
		return wid2
	while (prdcs[wid1, prdc] != -9999):
		if (r_pos.match(words['pos'][prdc]) or r_evntoprt.findall(','.join(words['evnt_annotp'][prdc])) or r_patn.findall(words['str'][prdc]) or r_patn.findall(words['stem'][prdc]) or r_pos.match(words['stem_pos'][prdc])):
			trigger = prdc
			break
		if (prdcs[wid1, prdc] == wid1):
			break
		prdc = prdcs[wid1,prdc]
	if (trigger is None):
		print 'Trigger not found between (%s, %s)' % (words['str'][wid1], words['str'][wid2])
	return trigger
	
	
def find_trgs(words, wid1, wid2, dist_mt, prdcs, source='2016', task='bb', pos_type='universal'):
	global POS_WEIGHT
	r_pos = re.compile(r'|'.join(POS_WEIGHT[pos_type].keys()), flags=re.I | re.X)
	prdc, triggers = wid1, []
	while (prdcs[wid2, prdc] != -9999):
		if (r_pos.findall(words['pos'][prdc])):
			triggers.append(prdc)
		if (prdcs[wid2, prdc] == wid2):
			break
		prdc = prdcs[wid2,prdc]
	return triggers
	
	
def print_tokens(tokens, trg_lbs=None, annots=None):
	result = []
	for i, token in enumerate(tokens):
		if (trg_lbs[i].sum() == 0):
			token = '<%s>' % token
		if (len(annots[i]) > 0):
			token = '[%s]' % token
		result.append(token)
	return ' '.join(result)

	
def get_data_trgs(raw_data, from_file=None, dataset='train', source='2016', task='bb', fmt='npz', spfmt='csr', ft_type='binary', max_df=1.0, min_df=1, parser='spacy', db_name='mesh2016', db_type='LevelDB', store_path='store'):
	# Read from local files
	if (from_file):
		if (type(from_file) == bool):
			file_name = ('%s_X.npz' % dataset, '%s_Y.npz' % dataset) if (fmt == 'npz') else ('%s_X.csv' % dataset, '%s_Y.csv' % dataset)
		else:
			file_name = from_file
		if (dataset == 'test'):
			print 'Reading file: %s' % (file_name[0])
			return io.read_df(os.path.join(DATA_PATH, source, task, file_name[0]), with_idx=True, sparse_fmt=spfmt)
		print 'Reading file: %s and %s' % (file_name[0], file_name[1])
		if (fmt == 'npz'):
			return io.read_df(os.path.join(DATA_PATH, source, task, file_name[0]), with_idx=True, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, source, task, file_name[1]), with_idx=True, sparse_fmt=spfmt)
		else:
			return pd.read_csv(os.path.join(DATA_PATH, source, task, file_name[0]), index_col=0, encoding='utf8'), pd.read_csv(os.path.join(DATA_PATH, source, task, file_name[1]), index_col=0, encoding='utf8')
	if (source == '2011'):
		pos_type = 'bnlpst2011'
	elif (parser == 'spacy'):
		pos_type = 'universal'
	elif (parser == 'stanford'):
		pos_type = 'stanford'
	global NOUN_POS
	r_pos = re.compile(NOUN_POS[pos_type], flags=re.I | re.X)
	# from rdflib import Graph
	from bionlp.spider.sparql import MeSHSPARQL
	from bionlp.util import ontology
	# g = Graph(store=db_type, identifier=db_name)
	# g.open(store_path)
	# g = MeSHSPARQL()
	from bionlp.spider import metamap
	mm_wrapper = metamap.Wrapper()
	mm_wrapper.start_service([1 - x for x in mm_wrapper.status()])
	## Feature columns
	evnt_index, ft_sub_str, ft_trg_str, ft_obj_str, label = [[] for i in range(5)]
	ft_order = ['sub', 'trg', 'obj']
	ft_name = {'sub':'Subject', 'trg':'Triggers', 'obj':'Object'}
	ft_dic = {'sub':ft_sub_str, 'trg':ft_trg_str, 'obj':ft_obj_str}
	evnt_lb = EVNT_LB[source]
	vft_dic, evnt_stat = [{} for i in range(2)]
	## Extract features from raw data
	for docid, corpus, preprcs, events in zip(raw_data['docids'], raw_data['corpus'], raw_data['preprcs'], raw_data['evnts']):
		sent_bndry, words, annots, depends, coref = preprcs
		word_num, event_num = len(words['str']), len(events['id']) if dataset != 'test' else 0
		## Construct the dependcy matrix
		dpnd_mt = {}
		# Combine the dependcy graphs of all sentences in a document
		for i in xrange(len(depends)):
			offset = sent_bndry[i]
			coo = coo_matrix(depends[i].values)
			for r, c, v in zip(coo.row, coo.col, coo.data):
				dpnd_mt[(offset + r, offset + c)] = v
		# Coreference resolution using nlp
		word_per_sent = sent_bndry[1:].mean()
		for crf in coref:
			for pairs in crf:
				tkn_num_pair = pairs[0][4] - pairs[0][3], pairs[1][4] - pairs[1][3]
				# Firstly intra-connect the annotations
				for intra_pair in itertools.permutations(range(pairs[0][3], pairs[0][4]), 2):
					dpnd_mt[(sent_bndry[pairs[0][1]] + intra_pair[0], sent_bndry[pairs[0][1]] + intra_pair[1])] = 1
				for intra_pair in itertools.permutations(range(pairs[1][3], pairs[1][4]), 2):
					dpnd_mt[(sent_bndry[pairs[1][1]] + intra_pair[0], sent_bndry[pairs[1][1]] + intra_pair[1])] = 1
				dpnd_mt[(sent_bndry[pairs[0][1]] + pairs[0][2], sent_bndry[pairs[1][1]] + pairs[1][2])] = 1
				dpnd_mt[(sent_bndry[pairs[1][1]] + pairs[1][2], sent_bndry[pairs[0][1]] + pairs[0][2])] = 1
		# Coreference resolution using ontology
		noun_tkns_list, concepts, mesh2word, word2mesh = [[] for i in range(3)] + [[-1 for i in range(len(words['str']))]]
		for i, pos in enumerate(words['pos']):
			if (r_pos.findall(pos)):
				noun_tkns_list.append((i, words['str'][i]))
		noun_tkns = [ntkn[1] for ntkn in noun_tkns_list]
		concept_dict, error = mm_wrapper.parse(noun_tkns)
		cncpt_idx = 0
		for idx, cncpts in concept_dict.iteritems():
			wid = noun_tkns_list[idx][0]
			concepts.append(cncpts[0])
			mesh2word.append(wid)
			word2mesh[wid] = cncpt_idx
			cncpt_idx += 1
		for cncpt_pair in itertools.combinations(range(len(concepts)), 2):
			if (concepts[cncpt_pair[0]].semtypes == concepts[cncpt_pair[1]].semtypes):
				dpnd_mt[(mesh2word[cncpt_pair[0]], mesh2word[cncpt_pair[1]])] = 1
				dpnd_mt[(mesh2word[cncpt_pair[1]], mesh2word[cncpt_pair[0]])] = 1
		# tc_list = [concept.tree_codes.split(';')[0] for concept in concepts]
		# lb_list = [concept.preferred_name for concept in concepts]
		# fn_func = ontology.define_fn(g, vrtx_type='label')
		# ont_dist_mt = ontology.min_span_subgraph(g, lb_list, fn_func, max_length=1)
		# Merge with the dependcy matrix
		# for row, col, val in zip(ont_dist_mt.row, ont_dist_mt.col, ont_dist_mt.data):
			# row, col = mesh2word[row], mesh2word[col]
			# if (dpnd_mt.setdefault((row, col), val) > val):
				# dpnd_mt[(row, col)] = val
		# Calculate shortest path in the document-level dependency graph
		dmt = nlp.dpnd_trnsfm(dpnd_mt, (word_num, word_num))
		dist_mt, prdcs = shortest_path(dmt, directed=False, return_predecessors=True)
		## Construct pairs of entities, format: <sub_aid, obj_aid>:[(sub_aindex, sub_tokens), trigger_tokens, (obj_aindex, obj_tokens), labels]
		entity_pairs = OrderedDict()
		for e_sub, e_obj in itertools.permutations(range(len(annots['id'])), 2):
			# Find the triggers in the longest_path between these two entities
			idx_pair, longest_path = None, 0
			for x in annots['tkns'][e_sub]:
				for y in annots['tkns'][e_obj]:
					if (dist_mt[x, y] != np.inf and dist_mt[x, y] > longest_path):
						idx_pair = (x, y)
						longest_path = dist_mt[x, y]
			# The entity pairs that across two sentences
			if (idx_pair is None):
				# print 'Skip: <%s,%s> in %s@%s' % (annots['id'][e_sub], annots['id'][e_obj], docid, dataset)
				continue
			triggers = find_trgs(words, idx_pair[0], idx_pair[1], dist_mt, prdcs, pos_type=pos_type, source=source)
			# Eliminate the tokens of entity pair in the triggers
			for wid in annots['tkns'][e_sub] + annots['tkns'][e_obj]:
				try:
					triggers.remove(wid)
				except:
					pass
			entity_pairs[(annots['id'][e_sub], annots['id'][e_obj])] = [(e_sub, annots['tkns'][e_sub]), triggers, (e_obj, annots['tkns'][e_obj]), []]
		## Add labels to the entity pairs
		for i in xrange(event_num):
			try:
				entity_pairs[(events['oprnds'][i][0], events['oprnds'][i][1])][3].append(events['type'][i])
			except KeyError:
				print 'Key Error: %s in %s@%s' % (str((events['oprnds'][i][0], events['oprnds'][i][1])), docid, dataset)
				idx_pair = annots['id'].index(events['oprnds'][i][0]), annots['id'].index(events['oprnds'][i][1])
				sent_idx_pair = (bisect(sent_bndry, annots['tkns'][idx_pair[0]][0]) - 1, bisect(sent_bndry, annots['tkns'][idx_pair[1]][0]) - 1)
				if (sent_idx_pair[0] != sent_idx_pair[1]):
					print 'The event cross sentences: %s' % str((sent_idx_pair[0], sent_idx_pair[1]))
					words0 = words['str'][sent_bndry[sent_idx_pair[0]]:sent_bndry[sent_idx_pair[0] + 1]] if sent_idx_pair[0] + 1 < len(sent_bndry) else words['str'][sent_bndry[-1]:]
					words1 = words['str'][sent_bndry[sent_idx_pair[1]]:sent_bndry[sent_idx_pair[1] + 1]] if sent_idx_pair[1] + 1 < len(sent_bndry) else words['str'][sent_bndry[-1]:]
					print '[%s] ' % annots['str'][idx_pair[0]] + ' '.join(words0) + '\n' + '[%s] ' % annots['str'][idx_pair[1]] + ' '.join(words1)
		## Extract the features from the preprocessed data
		for k, v in entity_pairs.iteritems():
			evnt_index.append((docid, k[0], k[1]))
			ft_sub_str.append(annots['str'][v[0][0]])
			ft_trg_str.append(' '.join([words['str'][wid] for wid in v[1]]))
			ft_obj_str.append(annots['str'][v[2][0]])
			label.append(v[3])
	# g.close()
	mm_wrapper.stop_service()
	
	## Convert the raw features into binary features
	Vectorizer = TfidfVectorizer if ft_type == 'tfidf' else CountVectorizer
	sub_vctrz, trg_vctrz, obj_vctrz = [Vectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', lowercase=False, max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False) for i in range(3)]
	vctrz_dic = dict(zip(ft_order, [sub_vctrz, trg_vctrz, obj_vctrz]))
	for fset in ft_order:
		ft_mt = vctrz_dic[fset].fit_transform(ft_dic[fset]).tocsr()
		classes = [cls[0] for cls in sorted(vctrz_dic[fset].vocabulary_.items(), key=operator.itemgetter(1))]
		vft_dic[fset] = (ft_mt, classes)
		
	## Label Construction
	if (dataset != 'test'):
		mlb = MultiLabelBinarizer()
		bin_label = (mlb.fit_transform(label), mlb.classes_)
	
	## Generate the features as well as the labels to form a completed dataset
	feat_mt = sp.sparse.hstack([vft_dic[fset][0] for fset in ft_order])
	feat_cols = ['%s_%s' % (fset, w) for fset in ft_order for w in vft_dic[fset][1]]
	feat_df = pd.DataFrame(feat_mt.todense(), index=evnt_index, columns=feat_cols)
	if (dataset != 'test'):
		label_df = pd.DataFrame(bin_label[0], index=evnt_index, columns=[tuple([lb] + EVNT_ARG_TYPE[source][lb]) for lb in bin_label[1]], dtype='int8')

	## Sampling
	obj_samp_idx = np.random.random_integers(0, feat_df.shape[0] - 1, size=200).tolist()
	ft_samp_idx = np.random.random_integers(0, feat_df.shape[1] - 1, size=1000).tolist()
	samp_feat_df = feat_df.iloc[obj_samp_idx, ft_samp_idx]
	if (dataset != 'test'):
		samp_lb_df = label_df.iloc[obj_samp_idx,:]

	## Output the dataset
	if (fmt == 'npz'):
		io.write_df(feat_df, os.path.join(DATA_PATH, source, task, '%s_X.npz' % dataset), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(samp_feat_df, os.path.join(DATA_PATH, source, task, '%s_sample_X.npz' % dataset), with_idx=True, sparse_fmt=spfmt, compress=True)
		if (dataset != 'test'):
			io.write_df(label_df, os.path.join(DATA_PATH, source, task, '%s_Y.npz' % dataset), with_idx=True, sparse_fmt=spfmt, compress=True)
			io.write_df(samp_lb_df, os.path.join(DATA_PATH, source, task, '%s_sample_Y.npz' % dataset), with_idx=True, sparse_fmt=spfmt, compress=True)
	else:
		feat_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_X.csv' % dataset), encoding='utf8')
		samp_feat_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_sample_X.csv' % dataset), encoding='utf8')
		if (dataset != 'test'):
			label_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_Y.csv' % dataset), encoding='utf8')
			samp_lb_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_sample_Y.csv' % dataset), encoding='utf8')
	if (dataset != 'test'):
		return feat_df, label_df
	else:
		return feat_df

		
def get_data_trg(raw_data, from_file=None, dataset='train', source='2016', task='bb', fmt='npz', spfmt='csr', ft_type='binary', max_df=1.0, min_df=1, parser='spacy'):
	# Read from local files
	if (from_file):
		if (type(from_file) == bool):
			word_X_name, word_y_name, edge_X_name, edge_y_name = ('%swX.npz' % dataset, '%swY.npz' % dataset, '%seX.npz' % dataset, '%seY.npz' % dataset) if (fmt == 'npz') else ('%sw_X.csv' % dataset, '%sw_Y.csv' % dataset, '%se_X.csv' % dataset, '%se_Y.csv' % dataset)
		else:
			word_X_name, word_y_name, edge_X_name, edge_y_name = from_file
		if (dataset == 'test'):
			print 'Reading file: %s, %s' % (word_X_name, 'test_rawdata.pkl')
			return io.read_df(os.path.join(DATA_PATH, source, task, word_X_name), with_idx=True, sparse_fmt=spfmt), io.read_obj(os.path.join(DATA_PATH, source, task, 'test_rawdata.pkl'))
		print 'Reading file: %s, %s, %s, %s' % (word_X_name, word_y_name, edge_X_name, edge_y_name)
		if (fmt == 'npz'):
			return io.read_df(os.path.join(DATA_PATH, source, task, word_X_name), with_idx=True, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, source, task, word_y_name), with_idx=True, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, source, task, edge_X_name), with_idx=True, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, source, task, edge_y_name), with_idx=True, sparse_fmt=spfmt)
		else:
			return pd.read_csv(os.path.join(DATA_PATH, source, task, word_X_name), index_col=0, encoding='utf8'), pd.read_csv(os.path.join(DATA_PATH, source, task, word_y_name), index_col=0, encoding='utf8'), pd.read_csv(os.path.join(DATA_PATH, source, task, edge_X_name), index_col=0, encoding='utf8'), pd.read_csv(os.path.join(DATA_PATH, source, task, edge_y_name), index_col=0, encoding='utf8')
			
	global ft_offset
	idx_range, evnt_lb, dist_mt_list, r_pos = [0, 0], EVNT_LB[source], [], re.compile('.*V.*')
	## Feature columns
	# Token features
	[ft_str, ft_cap, ft_pun, ft_digit, ft_stem, ft_pos, ft_btgram] = [[] for x in range(7)]
	# Annotation features
	[ft_annotp, ft_evntp, ft_evntoprt, ft_evntoprd, ft_trigger, edges, ft_edgelb] = [[] for x in range(7)]
	ft_order = ['has_cap', 'has_pun', 'has_digit', 'stem', 'pos', 'bt_gram']
	ft_name = ['Has Capital Letter', 'Has Punctuation', 'Has Digit', 'Stem', 'POS', 'Bi-Tri-Gram']
	ft_dic = {'has_cap':ft_cap, 'has_pun':ft_pun, 'has_digit':ft_digit, 'stem':ft_stem, 'pos':ft_pos, 'bt_gram':ft_btgram}
	
	# Statistical features
	[bft_dic, stem_frqs, pos_frqs, spos_frqs, annotp_frqs, annoth_frqs, evntp_frqs, evntoprt_frqs, evntoprd_frqs, dpnd_mt] = [{} for x in range(10)]
	# Extract information from raw data
	for docid, corpus, preprcs, events in zip(raw_data['docids'], raw_data['corpus'], raw_data['preprcs'], raw_data['evnts']):
		# print docid
		sent_bndry, words, annots, depends, coref = preprcs
		word_num = len(words['str'])
		# print ' '.join(words['str'])
		# Reset and record the word ID range for each document
		idx_range[0], idx_range[1] = (idx_range[1], idx_range[1] + word_num)
		ft_offset[dataset].append(idx_range[0])
		## Construct the token feature columns
		ft_str.extend(words['str'])
		ft_cap.extend([1 if any(str.isupper(c) for c in str(w)) else 0 for w in words['str']])
		ft_pun.extend([1 if any(c in string.punctuation for c in str(w)) else 0 for w in words['str']])
		ft_digit.extend([1 if any(c.isdigit() for c in str(w)) else 0 for w in words['str']])
		ft_stem.extend(words['stem'])
		# Construct the part-of-speech feature column
		ft_pos.extend(words['pos'])

		## Construct the annotation feature column
		ft_annotp.extend([[annots['type'][y] for y in x] if len(x) > 0 else [] for x in words['annot_id']])

		## Feature statistics

		## Construct the dependcy matrix
		sub_dpnd_mt = {}
		for i in xrange(len(depends)):
			offset = sent_bndry[i]
			coo = coo_matrix(depends[i].values)
			for r, c, v in zip(coo.row, coo.col, coo.data):
				dpnd_mt[(idx_range[0] + offset + r, idx_range[0] + offset + c)] = v
				sub_dpnd_mt[(offset + r, offset + c)] = v
		sdmt = nlp.dpnd_trnsfm(sub_dpnd_mt, (word_num, word_num))
		dist_mt, prdcs = shortest_path(sdmt, directed=False, return_predecessors=True)
		dist_mt_list.append(dist_mt)
		dist_mts[tuple(idx_range)] = dist_mt
		prdcss[tuple(idx_range)] = prdcs
		if (dataset == 'test'):
			continue
		## Construct the trigger feature column
		ft_trigger.extend(['' for x in range(word_num)])
		## Construct the event type feature column
		ft_evntp.extend(events['type'])
		event_num = len(events['id'])
		# Connect the event operands to the corresponding words and annotate the trigger
		words['evnt_annotp'] = [set([]) for x in words['str']] # Annotation types of tokens in events
		for i in xrange(event_num):
			annot_id_pair = annots['id'].index(events['oprnds'][i][0]), annots['id'].index(events['oprnds'][i][1])
			words_pair = annots['tkns'][annot_id_pair[0]], annots['tkns'][annot_id_pair[1]]
			for lw in words_pair[0]:
				words['evnt_annotp'][lw].add(events['oprnd_tps'][i][0])
			for rw in words_pair[1]:
				words['evnt_annotp'][rw].add(events['oprnd_tps'][i][1])
			idx_pair = (words_pair[0][0], words_pair[1][-1])
			events['oprnd_words'][i][0].extend(idx_range[0] + wid for wid in words_pair[0])
			events['oprnd_words'][i][1].extend(idx_range[0] + wid for wid in words_pair[1])
			# Find the triggers in the longest_path between these two entities
			idx_pair, longest_path = None, 0
			for x in xrange(len(words_pair[0])):
				for y in xrange(-1, -len(words_pair[1])-1, -1):
					if (dist_mt[x, y] > longest_path):
						idx_pair = (x, y)
			trigger = _find_trg(words, idx_pair[0], idx_pair[1], dist_mt, prdcs, source=source)
			if (trigger is not None):
				ft_trigger[idx_range[0] + trigger] = events['type'][i]
				# Generate the positive training samples, format: <src_wid, tgt_wid, distance>
				if (trigger == idx_pair[0]):
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[1], dist_mt[trigger, idx_pair[1]]))
					ft_edgelb.append(events['type'][i])
				elif (trigger == idx_pair[1]):
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[0], dist_mt[trigger, idx_pair[0]]))
					ft_edgelb.append(events['type'][i])
				else:
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[0], dist_mt[trigger, idx_pair[0]]))
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[1], dist_mt[trigger, idx_pair[1]]))
					ft_edgelb.extend([events['type'][i]]*2)
			# Find non-trigger verb
			for wid in xrange(len(words['str'])):
				if ((r_pos.match(words['pos'][wid]) or r_pos.match(words['stem_pos'][wid])) and ft_trigger[idx_range[0] + wid] != trigger):
					# Generate the negative training samples, format: <src_wid, tgt_wid, distance>
					if (wid == idx_pair[0]):
						edges.append((idx_range[0] + wid, idx_range[0] + idx_pair[1], dist_mt[wid, idx_pair[1]]))
						ft_edgelb.append('')
					elif (wid == idx_pair[1]):
						edges.append((idx_range[0] + wid, idx_range[0] + idx_pair[0], dist_mt[wid, idx_pair[0]]))
						ft_edgelb.append('')
					else:
						edges.append((idx_range[0] + wid, idx_range[0] + idx_pair[0], dist_mt[wid, idx_pair[0]]))
						edges.append((idx_range[0] + wid, idx_range[0] + idx_pair[1], dist_mt[wid, idx_pair[1]]))
						ft_edgelb.extend(['', ''])
		# Construct the event operands type column
		ft_evntoprt.extend(events['oprnd_tps'])
		# Event statistics
		for eventp in events['type']:
			evntp_frqs[eventp] = evntp_frqs.setdefault(eventp, 0) + 1
		for evntoprts in events['oprnd_tps']:
			evntoprt_frqs[evntoprts[0]] = evntoprt_frqs.setdefault(evntoprts[0], 0) + 1
			evntoprt_frqs[evntoprts[1]] = evntoprt_frqs.setdefault(evntoprts[1], 0) + 1

	## Construct word matrix
	for fset in ['has_cap', 'has_pun', 'has_digit']:
		bft_dic[fset] = (np.array(ft_dic[fset]).reshape(len(ft_dic[fset]),1), fset)
	for fset in ['stem', 'pos']:
		ft_classes = list(set(ft_dic[fset]))
		ft_mt = label_binarize(ft_dic[fset], classes=ft_classes)
		bft_dic[fset] = (ft_mt, ft_classes)
	# Construct the character bi-grams and trigrams
	btgv = CountVectorizer(ngram_range=(2, 3), analyzer='char', max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False)
	ft_chbtgram = btgv.fit_transform(ft_str).todense()
	ft_classes = [cls[0] for cls in sorted(btgv.vocabulary_.items(), key=operator.itemgetter(1))]
	bft_dic['bt_gram'] = (ft_chbtgram, ft_classes)
	# ft_annotp_mt = label_binarize(ft_annotp, classes=list(set(ft_annotp)))
	word_mt = np.hstack([bft_dic[fset][0] for fset in ft_order])
	wm_cols = ft_order[0:3] + ['%s_%s' % (fset, w) for fset in ft_order[3:] for w in bft_dic[fset][1]]
	word_df = pd.DataFrame(word_mt, columns=wm_cols)
	if (fmt == 'npz'):
		io.write_df(word_df, os.path.join(DATA_PATH, source, task, '%swX.npz'%dataset), sparse_fmt=spfmt, compress=True)
	else:
		word_df.to_csv(os.path.join(DATA_PATH, source, task, '%swX.csv'%dataset), encoding='utf8')

	if (dataset == 'test'):
		# Save intermediate data
		raw_data['word_offset'] = ft_offset[dataset]
		raw_data['dist_mts'] = dist_mt_list
		io.write_obj(raw_data, fpath=os.path.join(DATA_PATH, source, task, 'test_rawdata.pkl'))
		return word_df, raw_data
	else:
		# Construct trigger label
		trg_label = label_binarize(ft_trigger, classes=evnt_lb if len(evnt_lb) > 1 else ['']+evnt_lb)
		# print print_tokens(ft_str, trg_label, ft_annotp)
		# Construct trigger-argument pair sample matrix
		edge_data = np.array(edges)
		edge_mt = np.hstack((word_mt[edge_data[:,0].astype(int),:], word_mt[edge_data[:,1].astype(int),:]))
		em_cols = ['lf_%s' % col for col in wm_cols] + ['rt_%s' % col for col in wm_cols]
		# Combine all the data into Pandas DataFrame
		trg_lb = pd.DataFrame(trg_label, columns=evnt_lb)
		edge_df = pd.DataFrame(edge_mt, columns=em_cols)
		edge_lb_mt = label_binarize(ft_edgelb, classes=evnt_lb if len(evnt_lb) > 1 else ['']+evnt_lb)
		edge_lb = pd.DataFrame(edge_lb_mt, columns=evnt_lb)
		
		if (fmt == 'npz'):
			io.write_df(trg_lb, os.path.join(DATA_PATH, source, task, '%swY.npz'%dataset), sparse_fmt=spfmt, compress=True)
			io.write_df(edge_df, os.path.join(DATA_PATH, source, task, '%seX.npz'%dataset), sparse_fmt=spfmt, compress=True)
			io.write_df(edge_lb, os.path.join(DATA_PATH, source, task, '%seY.npz'%dataset), sparse_fmt=spfmt, compress=True)
		else:
			trg_lb.to_csv(os.path.join(DATA_PATH, source, task, '%swY.csv'%dataset), encoding='utf8')
			edge_df.to_csv(os.path.join(DATA_PATH, source, task, '%seX.csv'%dataset), encoding='utf8')
			edge_lb.to_csv(os.path.join(DATA_PATH, source, task, '%seY.csv'%dataset), encoding='utf8')

		return word_df, trg_lb, edge_df, edge_lb
		
## End Trigger-based Approach ##
  

## Start Non-Trigger Approach ##

def get_data_cbow(raw_data, from_file=None, ret_field='all', iterator=False, batch_size=32, dataset='train', source='2016', task='bb', fmt='npz', spfmt='csr', w2v_path='wordvec.bin', window_size=10, maxlen=None, npg_ratio=1.0):
	# Read from local files
	if (from_file):
		if (type(from_file) == bool):
			if (ret_field == 'all'):
				file_name = (['%s_X%i.%s' % (dataset, i, fmt) for i in range(4)] + ['%s_ent_X%i.%s' % (dataset, i, fmt) for i in range(2)] + ['%s_pseudo_X%i.%s' % (dataset, i, fmt) for i in range(2)], ['%s_Y.%s' % (dataset, fmt), '%s_ent_Y.%s' % (dataset, fmt)])
			elif (ret_field == 'event'):
				file_name = (['%s_X%i.%s' % (dataset, i, fmt) for i in range(4)], '%s_Y.%s' % (dataset, fmt))
			elif (ret_field == 'entity'):
				file_name = (['%s_ent_X%i.%s' % (dataset, i, fmt) for i in range(2)], '%s_ent_Y.%s' % (dataset, fmt))
			if (fmt == 'h5'):
				x_fname = ['cbow/%s' % os.path.splitext(xfn)[0] for xfn in file_name[0]]
				y_fname = 'cbow/%s' % os.path.splitext(file_name[1])[0] if (type(file_name[1]) != list) else ['cbow/%s' % os.path.splitext(yfn)[0] for yfn in file_name[1]]
				file_name = x_fname, y_fname, 'dataset.h5'
			# file_name = (['%s_X%i.%s' % (dataset, i, fmt) for i in range(4)], '%s_Y.%s' % (dataset, fmt)) if (fmt != 'h5') else (['cbow/%s_X%i' % (dataset, i) for i in range(4)], 'cbow/%s_Y' % dataset, 'dataset.h5')
		else:
			file_name = from_file
		if (dataset == 'test'):
			print 'Reading files: %s' % (', '.join(file_name[0]))
			if (fmt == 'npz'):
				return [io.read_df(os.path.join(DATA_PATH, source, task, fname), with_idx=True, sparse_fmt=None) for fname in file_name[0]], io.read_obj(os.path.join(DATA_PATH, source, task, 'test_rawdata.pkl'))
			elif (fmt == 'h5'):
				return [pd.read_hdf(os.path.join(DATA_PATH, source, task, file_name[2]), key=fname, iterator=iterator, chunksize=batch_size if iterator else None) for fname in file_name[0]], io.read_obj(os.path.join(DATA_PATH, source, task, 'test_rawdata.pkl'))
			else:
				return [pd.read_csv(os.path.join(DATA_PATH, source, task, fname), index_col=0, encoding='utf8') for fname in file_name[0]], io.read_obj(os.path.join(DATA_PATH, source, task, 'test_rawdata.pkl'))
		print 'Reading files: %s and %s' % (', '.join(file_name[0]), file_name[1])
		if (fmt == 'npz'):
			return [io.read_df(os.path.join(DATA_PATH, source, task, fname), with_idx=True, sparse_fmt=None) for fname in file_name[0]], io.read_df(os.path.join(DATA_PATH, source, task, file_name[1]), with_idx=True, sparse_fmt=spfmt) if (type(file_name[1]) != list) else [io.read_df(os.path.join(DATA_PATH, source, task, y), with_idx=True, sparse_fmt=spfmt) for y in file_name[1]]
		elif (fmt == 'h5'):
			return [pd.read_hdf(os.path.join(DATA_PATH, source, task, file_name[2]), key=fname, iterator=iterator, chunksize=batch_size if iterator else None) for fname in file_name[0]], pd.read_hdf(os.path.join(DATA_PATH, source, task, file_name[2]), key=file_name[1], iterator=iterator, chunksize=batch_size if iterator else None) if (type(file_name[1]) != list) else [pd.read_hdf(os.path.join(DATA_PATH, source, task, file_name[2]), key=y, iterator=iterator, chunksize=batch_size if iterator else None) for y in file_name[1]]
		else:
			return [pd.read_csv(os.path.join(DATA_PATH, source, task, fname), index_col=0, encoding='utf8') for fname in file_name[0]], pd.read_csv(os.path.join(DATA_PATH, source, task, file_name[1]), index_col=0, encoding='utf8') if (type(file_name[1]) != list) else [pd.read_csv(os.path.join(DATA_PATH, source, task, y), index_col=0, encoding='utf8') for y in file_name[1]]
	from bionlp.model import vecomnet
	w2v_wrapper = w2v.GensimW2VWrapper(w2v_path)
	last_widx = w2v_wrapper.get_vocab_size() - 1
	## Feature columns
	evnt_index, cbow, annot_cbow, direction, label, entity_label = [[] for i in range(6)]
	stat_oprnd_annot = {}
	## Extract features from raw data
	for docid, corpus, preprcs, events in itertools.izip(raw_data['docids'], raw_data['corpus'], raw_data['preprcs'], raw_data['evnts']):
		sent_bndry, words, annots, depends, coref = preprcs
		word_num, event_num = len(words['str']), len(events['id']) if dataset != 'test' else 0
		words['embedding_id'] = [w2v_wrapper.word2idx(w, inexistence=last_widx) for w in words['str']]
		annots['embedding_id'] = [w2v_wrapper.word2idx(w, inexistence=last_widx) for w in annots['type']]
		## Construct pairs of entities, format: <sub_aid, obj_aid>:[(sub_aindex, obj_aindex), directions, labels, (sub_label, obj_label)]
		entity_pairs = OrderedDict([((annots['id'][e_sub], annots['id'][e_obj]), [(e_sub, e_obj), [], [], []]) for e_sub, e_obj in itertools.permutations(range(len(annots['id'])), 2)])
		ok_samples = []
		## Add labels to the entity pairs
		for i in xrange(event_num):
			try:
				entity_pairs[(events['oprnds'][i][0], events['oprnds'][i][1])][1].append(1)
				entity_pairs[(events['oprnds'][i][0], events['oprnds'][i][1])][2].append(events['type'][i])
				entity_pairs[(events['oprnds'][i][0], events['oprnds'][i][1])][3] = [events['oprnd_tps'][i][0], events['oprnd_tps'][i][1]]
				entity_pairs[(events['oprnds'][i][1], events['oprnds'][i][0])][1].append(-1)
				entity_pairs[(events['oprnds'][i][1], events['oprnds'][i][0])][2].append(events['type'][i])
				entity_pairs[(events['oprnds'][i][1], events['oprnds'][i][0])][3] = [events['oprnd_tps'][i][1], events['oprnd_tps'][i][0]]
				# Inconsistent annotation
				if (EVNT_ARG_TYPE[source][events['type'][i]][0] != events['oprnd_tps'][i][0] or EVNT_ARG_TYPE[source][events['type'][i]][1] != events['oprnd_tps'][i][1]):
					print 'Event type %s has argument alias: %s %s!' % (events['type'][i], events['oprnd_tps'][i][0], events['oprnd_tps'][i][1])
					entity_pairs[(events['oprnds'][i][0], events['oprnds'][i][1])][3] = EVNT_ARG_TYPE[source][events['type'][i]]
					entity_pairs[(events['oprnds'][i][1], events['oprnds'][i][0])][3] = EVNT_ARG_TYPE[source][events['type'][i]][::-1]
				ok_samples.extend([events['oprnds'][i], events['oprnds'][i][::-1]])
				idx_pair = annots['id'].index(events['oprnds'][i][0]), annots['id'].index(events['oprnds'][i][1])
				stat_type = stat_oprnd_annot.setdefault(events['type'][i], [{},{}])
				stat_type[0].setdefault(annots['type'][idx_pair[0]], 0)
				stat_type[1].setdefault(annots['type'][idx_pair[1]], 0)
				stat_type[0][annots['type'][idx_pair[0]]] += 1
				stat_type[1][annots['type'][idx_pair[1]]] += 1
			except KeyError:
				print 'Key Error: %s in %s@%s' % (str((events['oprnds'][i][0], events['oprnds'][i][1])), docid, dataset)
				idx_pair = annots['id'].index(events['oprnds'][i][0]), annots['id'].index(events['oprnds'][i][1])
				sent_idx_pair = (bisect(sent_bndry, annots['tkns'][idx_pair[0]][0]) - 1, bisect(sent_bndry, annots['tkns'][idx_pair[1]][0]) - 1)
				if (sent_idx_pair[0] != sent_idx_pair[1]):
					print 'The event cross sentences: %s' % str((sent_idx_pair[0], sent_idx_pair[1]))
					words0 = words['str'][sent_bndry[sent_idx_pair[0]]:sent_bndry[sent_idx_pair[0] + 1]] if sent_idx_pair[0] + 1 < len(sent_bndry) else words['str'][sent_bndry[-1]:]
					words1 = words['str'][sent_bndry[sent_idx_pair[1]]:sent_bndry[sent_idx_pair[1] + 1]] if sent_idx_pair[1] + 1 < len(sent_bndry) else words['str'][sent_bndry[-1]:]
					print '[%s] ' % annots['str'][idx_pair[0]] + ' '.join(words0) + '\n' + '[%s] ' % annots['str'][idx_pair[1]] + ' '.join(words1)
		ng_all_samples = [k for k, v in entity_pairs.iteritems() if len(v[1]) == 0]
		ng_allsamp_num = len(ng_all_samples)
		if (dataset == 'test'):
			output_samples = ng_all_samples
			samp_idx = range(ng_allsamp_num)
		else:
			# Sampling the negative samples
			ng_size = int(1.0 * npg_ratio * max(max(1, len(annots['id'])/2), (len(entity_pairs) - ng_allsamp_num)))
			ng_sample_idx = np.random.choice(ng_allsamp_num, size=ng_size, replace=True if ng_allsamp_num < ng_size else False) if ng_allsamp_num>0 else []
			ng_samples = []
			for i in ng_sample_idx:
				ng_samples.append(ng_all_samples[i])
			output_samples = ok_samples + ng_samples
			samp_idx = range(len(output_samples))
			# np.random.shuffle(samp_idx)
		for i in samp_idx:
			e_sub, e_obj = entity_pairs[output_samples[i]][0]
			cbow_list = vecomnet.get_cbow_context(words['embedding_id'], [annots['tkns'][e_sub], annots['tkns'][e_obj]], window_size=window_size, include_target=True)
			# annot_cbow_list = vecomnet.get_cbow_context(annots['embedding_id'], [e_sub, e_obj], window_size=window_size/2, include_target=True)
			cbow_list = cbow_list[0] + cbow_list[1]
			for cl, annot_eid in zip(cbow_list, [annots['embedding_id'][e_sub]] * 2 + [annots['embedding_id'][e_obj]] * 2):
				cl.append(annot_eid)
			# annot_cbow_list = annot_cbow_list[0] + annot_cbow_list[1]
			evnt_index.append('|'.join((docid, output_samples[i][0], output_samples[i][1])))
			cbow.append(cbow_list)
			# annot_cbow.append(annot_cbow_list)
			direction.append(entity_pairs[output_samples[i]][1])
			label.append(entity_pairs[output_samples[i]][2])
			entity_label.append(entity_pairs[output_samples[i]][3])
	if (len(stat_oprnd_annot) > 0): io.write_obj(stat_oprnd_annot, os.path.join(DATA_PATH, source, task, '%s_stat_oprnd_annot' % dataset))
	## Feature Construction
	from keras.preprocessing import sequence
	cbow_seqs = [sequence.pad_sequences(x, maxlen=maxlen, dtype='int64', padding='pre', truncating='pre', value=last_widx) for x in zip(*cbow)]
	feat_dfs = [pd.DataFrame(mt, index=evnt_index, columns=['cbow_%i'%i for i in range(mt.shape[1])], dtype='int64') for mt in cbow_seqs]
	for i, df in enumerate(feat_dfs):
		if (fmt == 'npz'):
			io.write_df(df, os.path.join(DATA_PATH, source, task, '%s_X%i.npz' % (dataset, i)), sparse_fmt=None, compress=True)
		elif (fmt == 'h5'):
			df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_X%i' % (dataset, i), format='table', data_columns=True)
		else:
			df.to_csv(os.path.join(DATA_PATH, source, task, '%s_X%i.npz' % (dataset, i)), encoding='utf8')
	# Obtain the entity indices
	lent_index = ['|'.join(idx.split('|')[:2]) for idx in evnt_index]
	lent_feat_dfs = [x.set_index([lent_index]).reset_index().drop_duplicates(subset='index').set_index('index') for x in feat_dfs[:2]]
	rent_index = ['|'.join(idx.split('|')[::2]) for idx in evnt_index]
	rent_feat_dfs = [x.set_index([rent_index]).reset_index().drop_duplicates(subset='index').set_index('index') for x in feat_dfs[2:]]
	ent_feat_dfs = [pd.concat([lfdf, rfdf], axis=0).reset_index().drop_duplicates(subset='index').set_index('index') for lfdf, rfdf in zip(lent_feat_dfs, rent_feat_dfs)]
	## Label Construction
	if (dataset != 'test'):
		event_mlb = MultiLabelBinarizer()
		bin_label = (event_mlb.fit_transform(label), event_mlb.classes_.tolist())
		# Combine the binary labels and the directions
		for i in xrange(len(direction)):
			for j in range(len(direction[i])):
				idx = bin_label[1].index(label[i][j])
				bin_label[0][i][idx] = bin_label[0][i][idx] * direction[i][j]
		label_df = pd.DataFrame(bin_label[0], index=evnt_index, columns=[':'.join([lb] + EVNT_ARG_TYPE[source][lb]) for lb in bin_label[1]], dtype='int8')
		# Labels for entity types
		lent_label, rent_label = zip(*[x if x else ['Unknown']*2 for x in entity_label])
		# ent_label = lent_label
		ent_label = lent_label + rent_label
		ent_lbr = LabelBinarizer()
		ent_bin_label = (ent_lbr.fit_transform(ent_label).astype('int8'), ent_lbr.classes_.tolist())
		# ent_label_df = pd.DataFrame(ent_bin_label[0], index=lent_index, columns=ent_bin_label[1], dtype='int8').drop('Unknown', axis=1)
		ent_label_df = pd.DataFrame(ent_bin_label[0], index=lent_index + rent_index, columns=ent_bin_label[1], dtype='int8').drop('Unknown', axis=1)
		ent_label_cols = ent_label_df.columns
		ent_label_df = ent_label_df.groupby(ent_label_df.index).sum(axis=0) # deal with duplicated indices
		ent_label_df.columns = ent_label_cols
		ent_label_df[ent_label_df > 1] = 1
		ent_feat_dfs = [df.loc[ent_label_df.index] for df in ent_feat_dfs]
		# Combine the labels of both entity types in a event
		lent_idx, rent_idx = [], []
		for idx in evnt_index:
			docid, lent, rent = idx.split('|')
			lent_idx.append('|'.join([docid, lent]))
			rent_idx.append('|'.join([docid, rent]))
		lent_df = ent_label_df.loc[lent_idx].reset_index().drop('index', axis=1).astype('int8')
		rent_df = ent_label_df.loc[rent_idx].reset_index().drop('index', axis=1).astype('int8')
		# Combine into one matrix
		# ent_df = pd.concat([lent_df, rent_df], axis=1).astype('int8')
		# ent_df.index, ent_df.columns = evnt_index, ['LF-%s' % col if i < ent_label_df.shape[1] else 'RT-%s' % col for i, col in enumerate(ent_df.columns)]
		# Separate into two matrixes
		# lent_df.index = evnt_index
		# ent_df = lent_df
		lent_df.index, rent_df.index = evnt_index, evnt_index
		ent_dfs = [lent_df, rent_df]
		if (fmt == 'npz'):
			io.write_df(label_df, os.path.join(DATA_PATH, source, task, '%s_Y.npz' % dataset), sparse_fmt=spfmt, compress=True)
			io.write_df(ent_label_df, os.path.join(DATA_PATH, source, task, '%s_ent_Y.npz' % dataset), sparse_fmt=spfmt, compress=True)
			_ = [io.write_df(df, os.path.join(DATA_PATH, source, task, '%s_ent_X%i.npz' % (dataset, i)), sparse_fmt=None, compress=True) for i, df in enumerate(ent_feat_dfs)]
			# io.write_df(ent_df, os.path.join(DATA_PATH, source, task, '%s_pseudo_X.npz' % dataset), sparse_fmt=None, compress=True)
			_ = [io.write_df(df, os.path.join(DATA_PATH, source, task, '%s_pseudo_X%i.npz' % (dataset, i)), sparse_fmt=None, compress=True) for i, df in enumerate(ent_dfs)]
		elif (fmt == 'h5'):
			label_df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_Y' % dataset, format='table', data_columns=True)
			ent_label_df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_ent_Y' % dataset, format='table', data_columns=True)
			_ = [df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_ent_X%i' % (dataset, i), format='table', data_columns=True) for i, df in enumerate(ent_feat_dfs)]
			# ent_df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_pseudo_X' % dataset, format='table', data_columns=True)
			_ = [df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_pseudo_X%i' % (dataset, i), format='table', data_columns=True) for i, df in enumerate(ent_dfs)]
		else:
			label_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_Y.csv' % dataset), encoding='utf8')
			ent_label_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_ent_Y.csv' % dataset), encoding='utf8')
			_ = [df.to_csv(os.path.join(DATA_PATH, source, task, '%s_ent_X%i.csv' % (dataset, i)), encoding='utf8') for i, df in enumerate(ent_feat_dfs)]
			# ent_df.to_csv(os.path.join(DATA_PATH, source, task, '%s_pseudo_X.csv' % dataset), encoding='utf8')
			_ = [df.to_csv(os.path.join(DATA_PATH, source, task, '%s_pseudo_X%i.csv' % (dataset, i)), encoding='utf8') for i, df in enumerate(ent_dfs)]
		# if (ret_field == 'all'): return feat_dfs + ent_feat_dfs + [ent_df], [label_df, ent_label_df]
		if (ret_field == 'all'): return feat_dfs + ent_feat_dfs + ent_dfs, [label_df, ent_label_df]
		return feat_dfs, label_df if (ret_field == 'event') else ent_feat_dfs, ent_label_df
	else:
		io.write_obj(raw_data, fpath=os.path.join(DATA_PATH, source, task, 'test_rawdata.pkl'))
		for i, df in enumerate(ent_feat_dfs):
			if (fmt == 'npz'):
				io.write_df(df, os.path.join(DATA_PATH, source, task, '%s_ent_X%i.npz' % (dataset, i)), sparse_fmt=None, compress=True)
			elif (fmt == 'h5'):
				df.to_hdf(os.path.join(DATA_PATH, source, task, 'dataset.h5'), 'cbow/%s_ent_X%i' % (dataset, i), format='table', data_columns=True)
			else:
				df.to_csv(os.path.join(DATA_PATH, source, task, '%s_ent_X%i.csv' % (dataset, i)), encoding='utf8')
		if (ret_field == 'all'): return feat_dfs + ent_feat_dfs, raw_data
		return feat_dfs, raw_data if (ret_field == 'event') else ent_feat_dfs, raw_data

## End Non-Trigger Approach ##


## Reverse the predictions to raw data
def pred2data(preds, method='cbow', scheme='', **kwargs):
	if (method == 'trigger'):
		if (scheme == 'trgs'):
			return pred2data_trgs(preds, **kwargs)
		elif (scheme == 'trg'):
			return pred2data_trg(preds, **kwargs)
	elif (method == 'cbow'):
		return pred2data_cbow(preds, **kwargs)
	
		
def pred2data_cbow(preds, source='2016', task='bb'):
	events = {}
	for i, row in preds.iterrows():
		docid, lent, rent = row.name.split('|')
		for evnt_args, direction in row[row!=0].iteritems():
			evnt_type, lent_type, rent_type = evnt_args.split(':')
			if (direction == -1): lent_type, lent, rent_type, rent = rent_type, rent, lent_type, lent
			if (EVNT_ARG_TYPE[source][evnt_type][0] != lent_type or EVNT_ARG_TYPE[source][evnt_type][1] != rent_type): continue
			events.setdefault(docid, []).append((evnt_type, ':'.join([lent_type, lent]), ':'.join([rent_type, rent])))
	return events
			

def to_a2(events, dir_path='.', source='2016', task='bb'):
	if (source == '2016'):
		to_a2_2016(events, dir_path)
	elif (source == '2013'):
		to_a2_2016(events, dir_path)
	elif (source == '2011'):
		to_a2_2011(events, dir_path)
		
		
def to_a2_2011(events, dir_path='.'):
	fs.mkdir(dir_path)
	for docid, event_list in events.iteritems():
		content = '\n'.join(['E%i\t%s' % (i+1, x) for i, x in enumerate(list(set([' '.join(evnt) for evnt in event_list])))]) + '\n'
		fs.write_file(content, os.path.join(dir_path, '%s.a2' % docid))
		
		
def to_a2_2016(events, dir_path='.'):
	fs.mkdir(dir_path)
	for docid, event_list in events.iteritems():
		content = '\n'.join(['R%i\t%s' % (i+1, x) for i, x in enumerate(list(set([' '.join(evnt) for evnt in event_list])))]) + '\n'
		fs.write_file(content, os.path.join(dir_path, '%s.a2' % docid))