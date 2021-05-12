#!/usr/bin/env

from binascii import hexlify
from collections import Counter
from itertools import combinations, compress
from lexical_diversity import lex_div as ld
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn import svm, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer, LabelBinarizer
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from string import punctuation
from tqdm import tqdm
from tqdm import trange
import argparse
import colorsys
import glob
import heapq
import itertools
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pickle
import random
import re
import sys
import warnings

"""
PARAMETERS
"""
sample_len = 1300 # length of sample / segment
n_feats = 350 # number of features taken in to account
ignore_words = [] # Words in list will be ignored in analysis
list_of_function_words = open('params/fword_list.txt').read().split() # Loads manual list of function words

"""
GENERAL CLASSES AND FUNCTIONS
"""
def enclitic_split(input_str):
	# Feed string, returns lowercased text with split enclitic -que
	que_list = open("params/que_list.txt").read().split()
	spaced_text = []
	for word in input_str.split():
		word = "".join([char for char in word if char not in punctuation]).lower()
		if word[-3:] == 'que' and word not in que_list:
			word = word.replace('que','') + ' que'
		spaced_text.append(word)
	spaced_text = " ".join(spaced_text)
	return spaced_text

def words_and_bigrams(text):
	words = re.findall(r'\w{1,}', text)
	for w in words:
		if w not in stop_words:
			yield w.lower()
		for i in range(len(words) - 2):
			if ' '.join(words[i:i+2]) not in stop_words:
				yield ' '.join(words[i:i+2]).lower()

def to_dense(X):
		X = X.todense()
		X = np.nan_to_num(X)
		return X

def deltavectorizer(X):
	    # "An expression of pure difference is what we need"
	    #  Burrows' Delta -> Absolute Z-scores
	    X = np.abs(stats.zscore(X))
	    X = np.nan_to_num(X)
	    return X

def most_common(lst):
	return max(set(lst), key=lst.count)

def align_yaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)

def align_xaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	x1, _ = ax1.transData.transform((v1, 0))
	x2, _ = ax2.transData.transform((v2, 0))
	inv = ax2.transData.inverted()
	dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
	minx, maxx = ax2.get_xlim()
	ax2.set_xlim(minx+dx, maxx+dx)

def change_intensity(color, amount=0.5):
	"""
	Lightens the given color by multiplying (1-luminosity) by the given amount.
	Input can be matplotlib color string, hex string, or RGB tuple.

	Examples:
	>> change_intensity('g', 0.3)
	>> change_intensity('#F034A3', 0.6)
	>> change_intensity((.3,.55,.1), 0.5)
	https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

	setting an amount < 1 lightens
	setting an amount > 1 darkens too

	"""
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

class DataReader:
	"""
	Parameters 
	----------
	folder_location: location of .txt files in directory
	sample_len: declare sample length

	Returns
	-------
	Lists
	authors = [A, A, B, B, ...]
	titles = [A, A, B, B, ...]
	texts = [s, s, s, s, ...] # strings
	"""

	def __init__(self, folder_location, sample_len):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def fit(self, shingling, shingle_titles):

		authors = []
		titles = []
		texts = []

		if shingling == 'n':
		
			for filename in glob.glob(self.folder_location + "/*"):
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split("/")[-1].split(".")[0].split("_")[1]

				bulk = []
				text = open(filename).read()

				for word in text.strip().split():
					word = re.sub('\d+', '', word) # escape digits
					word = re.sub('[%s]' % re.escape(punctuation), '', word) # escape punctuation
					word = word.lower() # convert upper to lowercase
					bulk.append(word)

				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]
				bulk = [bulk[i:i+self.sample_len] for i \
					in range(0, len(bulk), self.sample_len)]
				for index, sample in enumerate(bulk):
					if len(sample) == self.sample_len:
						authors.append(author)
						titles.append(title + "_{}".format(str(index + 1)))
						texts.append(" ".join(sample))

		# titles which should be shingled can be fed
		if shingling == 'y':
			
			step_size = 10

			for filename in glob.glob(self.folder_location + "/*"):
				
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split("/")[-1].split(".")[0].split("_")[1]
				text = open(filename).read()

				text = re.sub('[%s]' % re.escape(punctuation), '', text) # Escape punctuation and make characters lowercase
				text = re.sub('\d+', '', text)
				text = text.lower().split()

				if title in shingle_titles:

					steps = np.arange(0, len(text), step_size)
					step_ranges = []
					data = {}
					for each_begin in steps:
						key = '{}-{}-{}'.format(title, str(each_begin), str(each_begin + sample_len))
						sample_range = range(each_begin, each_begin + sample_len)
						step_ranges.append(sample_range)
						sample = []
						for index, word in enumerate(text):
							if index in sample_range:
								sample.append(word)
						if len(sample) == sample_len:
							data[key] = sample

					for key, sample in data.items():
						authors.append(author)
						titles.append(key)
						texts.append(" ".join(sample))

				else:
					# Safety measure against empty strings in samples
					bulk = [word for word in text if word != ""]
					bulk = [bulk[i:i + sample_len] for i \
						in range(0, len(bulk), sample_len)]
					for index, sample in enumerate(bulk):
						if len(sample) == sample_len:
							authors.append(author)
							titles.append(title + "_{}".format(str(index + 1)))
							texts.append(" ".join(sample))

		return authors, titles, texts


class Vectorizer:
	"""
	Independent class to vectorize texts.

	Parameters
	---------
	"""

	def __init__(self, texts, stop_words, n_feats, feat_scaling, analyzer, vocab):
		self.texts = texts
		self.stop_words = stop_words
		self.n_feats = n_feats
		self.feat_scaling = feat_scaling
		self.analyzer = analyzer
		self.vocab = vocab
		self.norm_dict = {'delta': FunctionTransformer(deltavectorizer), 
						  'normalizer': Normalizer(),
						  'standard_scaler': StandardScaler()}

	# Raw Vectorization

	def raw(self):

		# Text vectorization; array reversed to order of highest frequency
		# Vectorizer takes a list of strings

		# Define fed-in analyzer
		ngram_range = None
		if self.analyzer == 'char':
			ngram_range = ((4,4))
		elif self.analyzer == 'word':
			ngram_range = ((1,1))

		"""option where only words from vocab are taken into account"""
		model = CountVectorizer(stop_words=self.stop_words, 
								max_features=self.n_feats,
								analyzer=self.analyzer,
								vocabulary=self.vocab,
								ngram_range=ngram_range)

		doc_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(doc_vectors, axis=0))
		
		""" ||| Input vocabulary retains original order, 
		new vocabulary is ordered in terms of frequency |||"""
		if self.vocab == None:
			features = model.get_feature_names()
			doc_features = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
		else:
			"""if a vocabulary is given, sort it in terms of freq nevertheless"""
			features = self.vocab
			doc_features = model.get_feature_names()
			doc_features = [feat for (freq, feat) in sorted(zip(corpus_vector, features), reverse=True)]
			"""only retain max number of n feats"""
			doc_features = doc_features[:self.n_feats]

		new_X = []
		for feat in doc_features:
			for ft, vec in zip(model.get_feature_names(), doc_vectors.transpose()):
				if feat == ft: 
					new_X.append(vec)
		new_X = np.array(new_X).transpose()
		doc_vectors = new_X

		if self.feat_scaling == False:
			scaling_model = None
			pass
		else:
			scaling_model = self.norm_dict[self.feat_scaling]
			doc_vectors = scaling_model.fit_transform(doc_vectors)

		return doc_vectors, doc_features, scaling_model

	# Term-Frequency Inverse Document Frequency Vectorization

	def tfidf(self, smoothing):

		# Define fed-in analyzer
		ngram_range = None
		stop_words = self.stop_words
		if self.analyzer == 'char':
			ngram_range = ((4,4))
		elif self.analyzer == 'word':
			ngram_range = ((1,1))

		model = TfidfVectorizer(stop_words=self.stop_words, 
								max_features=self.n_feats,
								analyzer=self.analyzer,
								vocabulary=self.vocab,
								ngram_range=ngram_range)

		tfidf_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(tfidf_vectors, axis=0))
		
		""" ||| Input vocabulary retains original order, 
		new vocabulary is ordered in terms of frequency |||"""
		if self.vocab == None:
			features = model.get_feature_names()
			tfidf_features = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
		else:
			"""if a vocabulary is given, sort it in terms of freq nevertheless"""
			features = self.vocab
			tfidf_features = model.get_feature_names()
			tfidf_features = [feat for (freq, feat) in sorted(zip(corpus_vector, features), reverse=True)]
			"""only retain max number of n feats"""
			tfidf_features = tfidf_features[:self.n_feats]

		new_X = []
		for feat in tfidf_features:
			for ft, vec in zip(model.get_feature_names(), tfidf_vectors.transpose()):
				if feat == ft: 
					new_X.append(vec)
		new_X = np.array(new_X).transpose()
		tfidf_vectors = new_X

		if self.feat_scaling == False:
			scaling_model = None
			pass
		else:
			scaling_model = self.norm_dict[self.feat_scaling]
			tfidf_vectors = scaling_model.fit_transform(tfidf_vectors)
			
		return tfidf_vectors, tfidf_features, scaling_model

class Phase_One:
	"""
	Class that runs a parameter search.

	Parameters
	---------
	folder_location = location of .txt files
	
	Returns
	-------
	optimal_sample_len
	optimal_feature_type
	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location

	def go(self):
		results_file = open('results.txt', 'a')

		sample_len_loop = list(range(50, 1350, 25))
		# feat_type_loop = ['raw_fwords','raw_MFW','raw_4grams','tfidf_fwords','tfidf_MFW','tfidf_4grams']
		feat_type_loop = ['tfidf_fwords']
		c_options = [1] # 10, 100 and 1000 also possible, but on average scores are worse

		for feat_type in feat_type_loop:
			# vector length has to differ according to feature type (since there are no 1,000 function words)
			if feat_type.split('_')[-1] == 'fwords':
				# feat_n_loop = [50, 150, 250, 300]
				feat_n_loop = [150]
			else:
				feat_n_loop = [250, 500, 750, 1000]
			
			for n_feats in feat_n_loop:
				for sample_len in sample_len_loop:
					# Leave One Out cross-validation
					"""
					PREPROCESSING
					-------------
					"""
					# Load training files
					# The val_1 and val_2 pass True or False arguments to the sampling method
					authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])
					invalid_words = []
						
					# Number of splits is based on number of samples, so only possible afterwards
					# Minimum is 2
					n_cval_splits = 10

					# Try both stratified cross-validation as 'normal' KFold cross-validation.
					# Stratification has already taken place with random sampling
					cv_types = []
					cv_types.append(StratifiedKFold(n_splits=n_cval_splits))

					"""
					ACTIVATE VECTORIZER
					"""
					if feat_type == 'raw_MFW': 
						vectorizer = CountVectorizer(stop_words=invalid_words, 
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)
					elif feat_type == 'tfidf_MFW': 
						vectorizer = TfidfVectorizer(stop_words=invalid_words, 
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)
					elif feat_type == 'raw_fwords':
						"""
						All content words of corpus are rendered invalid
						and fed in the model as stop_words
						"""
						stop_words = [t.split() for t in texts]
						stop_words = sum(stop_words, [])
						stop_words = [w for w in stop_words if w not in list_of_function_words]
						stop_words = set(stop_words)
						stop_words = list(stop_words)

						"""
						----
						"""
						vectorizer = CountVectorizer(stop_words=stop_words,
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)

					elif feat_type == 'tfidf_fwords': 
						"""
						Low-frequency function words gain higher weight
						Filters out words that are not function words
						"""
						stop_words = [t.split() for t in texts]
						stop_words = sum(stop_words, [])
						stop_words = [w for w in stop_words if w not in list_of_function_words]
						stop_words = set(stop_words)
						stop_words = list(stop_words)
						"""
						----
						"""
						vectorizer = TfidfVectorizer(stop_words=stop_words, 
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)

					elif feat_type == 'raw_4grams': 
						vectorizer = CountVectorizer(stop_words=invalid_words, 
													analyzer='char', 
													ngram_range=(4, 4),
													max_features=n_feats)

					elif feat_type == 'tfidf_4grams': 
						vectorizer = TfidfVectorizer(stop_words=invalid_words, 
													analyzer='char', 
													ngram_range=(4, 4),
													max_features=n_feats)
					
					"""
					ENCODING X_TRAIN, x_test AND Y_TRAIN, y_test
					--------------------------------------------
			 		"""
					# Arranging dictionary where title is mapped to encoded label
					# Ultimately yields Y_train

					label_dict = {}
					inverse_label = {}
					for title in authors: 
						label_dict[title.split('_')[0]] = 0 
					for i, key in zip(range(len(label_dict)), label_dict.keys()):
						label_dict[key] = i
						inverse_label[i] = key

					"""
					TRAINING

					Step 1: input string is vectorized
						e.g. '... et quam fulgentes estis in summo sole ...'
					Step 2: to_dense = make sparse into dense matrix
					Step 3: feature scaling = normalize frequencies to chosen standard
					Step 4: reduce dimensionality by performing feature selection
					Step 5: choose type of classifier with specific decision function

					"""
					# Map Y_train to label_dict

					Y_train = []
					for title in authors:
						label = label_dict[title.split('_')[0]]
						Y_train.append(label)

					Y_train = np.array(Y_train)
					X_train = texts

					# DECLARING GRID, TRAINING
					# ------------------------
					"""
					Put this block of code in comment when skipping training and loading model
					Explicit mention labels=Y_train in order for next arg average to work
					average='macro' denotes that precision-recall scoring (in principle always binary) 
					needs to be averaged for multi-class problem
					"""

					pipe = Pipeline(
						[('vectorizer', vectorizer),
						 ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
						 ('feature_scaling', StandardScaler()),
						 # ('reduce_dim', SelectKBest(mutual_info_regression)),
						 ('classifier', svm.SVC())])

					# c_options = [c_parameter]
					# n_features_options = [n_feats]
					kernel_options = ['linear']

					param_grid = [	
						{
							'vectorizer': [vectorizer],
							'feature_scaling': [StandardScaler()],
							# 'reduce_dim': [SelectKBest(mutual_info_regression)],
							# 'reduce_dim__k': n_selected_feats,
							'classifier__C': c_options,
							'classifier__kernel': kernel_options,
						},
					]

					# Change this parameter according to preferred high scoring metric
					refit = 'accuracy_score'

					for cv in cv_types:

						print(":::{} as feature type, {} as number of features ::::".format(feat_type, str(n_feats)))
						grid = GridSearchCV(pipe, cv=cv, n_jobs=9, param_grid=param_grid,
											scoring={
										 		'precision_score': make_scorer(precision_score, labels=Y_train, \
										 														 average='macro'),
												'recall_score': make_scorer(recall_score, labels=Y_train, \
												 										  average='macro'),
												'f1_score': make_scorer(f1_score, labels=Y_train, \
												 								  average='macro'),
												'accuracy_score': make_scorer(accuracy_score),},
											refit=refit, 
											# Refit determines which scoring method weighs through in model selection
											verbose=True
											# Verbosity level: amount of info during training
											) 

						# Get best model & parameters
						# Save model locally
						grid.fit(X_train, Y_train)
						model = grid.best_estimator_

						# Safety buffer: to avoid errors in code
						vectorizer = model.named_steps['vectorizer']
						classifier = model.named_steps['classifier']
						features = vectorizer.get_feature_names()
						best_c_param = classifier.get_params()['C']
						# features_booleans = grid.best_params_['reduce_dim'].get_support()
						# grid_features = list(compress(features, features_booleans))

						if len(features) != n_feats:
							sys.exit("ERROR: Inconsistent number of features: {} against {}".format(str(n_feats),str(len(features))))

						model_name = '{}-{}feats-{}w-c{}-model'.format(feat_type, str(n_feats), str(sample_len), str(best_c_param))
						model_location = 'results/models/{}-{}feats-{}w-c{}-model'.format(feat_type, str(n_feats), str(sample_len), str(best_c_param))
						pickle.dump(grid, open(model_location, 'wb'))

						accuracy = grid.cv_results_['mean_test_accuracy_score'][0]
						precision = grid.cv_results_['mean_test_precision_score'][0]
						recall = grid.cv_results_['mean_test_recall_score'][0]
						f1 = grid.cv_results_['mean_test_f1_score'][0]

						results_file.write(model_name + '\t' + str(accuracy))
						results_file.write('\n')

class Plot_Lowess:
	"""
	Class that outputs accuracy in function of a variable
	with fitted lowess line

	Parameters
	----------
	results_location: opens file containing model results
	
	Returns
	-------
	.pdf figure
	"""

	def __init__(self, results_location):
		self.results_location = results_location

	def plot(self):

		results_location = open('results/results.txt')
		sample_len_loop = list(range(50, 1350, 25))

		x = sample_len_loop
		# actual accuracies!
		y = [0.6302142702546714, 0.6462571662571663, 0.6509753298909925, 0.7146766169154228, 0.7163311688311688, 0.7263297872340425, 0.8035423925667828, 0.8222222222222222, 0.7902462121212122, 0.8275268817204303, 0.8359788359788359, 0.8235384615384616, 0.8327898550724637, 0.870995670995671, 0.8609523809523809, 0.8868421052631579, 0.8783625730994151, 0.9058823529411765, 0.8889705882352942, 0.8970833333333333, 0.9466666666666667, 0.8857142857142858, 0.9417582417582417, 0.9141025641025641, 0.9365384615384615, 0.9416666666666667, 0.9393939393939394, 0.9181818181818183, 0.9245454545454546, 0.9718181818181819, 0.9277777777777778, 0.95, 0.9477777777777778, 0.9788888888888888, 0.9666666666666666, 0.9666666666666666, 0.9277777777777778, 0.9416666666666667, 0.95, 0.95, 0.9625, 0.9625, 0.975, 0.9857142857142858, 0.9857142857142858, 0.9857142857142858, 0.9714285714285715, 0.9571428571428571, 0.9523809523809523, 0.9833333333333334, 0.9833333333333334, 0.9666666666666668]

		fig = plt.figure(figsize=(6,2))
		ax = fig.add_subplot(111)

		ys0 = lowess(y, x)
		lowess_x0 = ys0[:,0]
		lowess_y0 = ys0[:,1]

		for p1, p2 in zip(x, y):
			ax.scatter(p1, p2, marker='o', color='w', s=10, edgecolors='k', linewidth=0.5)

		ax.plot(lowess_x0, lowess_y0, color='k', linewidth=0.5, markersize=1.5, linestyle='--')

		rcParams['font.family'] = 'sans-serif'
		rcParams['font.sans-serif'] = ['Bodoni 72'] # Font of Revue Mabillon

		ax.set_xlabel('Sample length')
		ax.set_ylabel('Accuracy')

		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(7)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(7)

		# Despine
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(True)
		ax.spines['bottom'].set_visible(True)

		plt.tight_layout()
		plt.show()

		fig.savefig("figs/sample_len_accuracy.pdf", \
					transparent=True, format='pdf')

class PrinCompAnal:

	""" |--- Principal Components Analysis ---|
		::: Plots PCA Plot ::: """

	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def plot(self):

		invalid_words = []
		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='y', shingle_titles=['shingled-epistolas'])

		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='standard_scaler',
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		pca = PCA(n_components=3)
		
		X_bar = pca.fit_transform(X)
		var_exp = pca.explained_variance_ratio_

		var_pc1 = np.round(var_exp[0]*100, decimals=2)
		var_pc2 = np.round(var_exp[1]*100, decimals=2)
		var_pc3 = np.round(var_exp[2]*100, decimals=2)
		explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)
		comps = pca.components_
		comps = comps.transpose()
		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(features, comps[:,0]), \
								  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(features, comps[:,1]), \
						   		  key=lambda tup: tup[1], reverse=True)

		print("Explained variance: ", explained_variance)
		print("Number of words: ", len(features))
		print("Sample size : ", sample_len)
		

		rcParams['font.family'] = 'sans-serif'
		rcParams['font.sans-serif'] = ['Bodoni 72']

		legend_dictionary = {'Alanus-de-Insulis': 'Alan of Lille',
							 'Anselmus-Cantuariensis': 'Anselm of Canterbury'}
		customized_colors = {}
		customized_markers = {'Abbo-Floriacensis': 'o',
							 'Adso-Dervensis': 's',
							 'Ioannis-Sancti-Arnulfi': '^'}
		customized_alpha = {'Theodericus-Floriacensis': 0.8,
							'Beda-Venerabilis': 0.17,
							'Beda-Venerabilis': 0.17,
							'Abbo-Floriacensis': 0.17,
							'Adso-Dervensis': 0.4,
							'Helgaldus-Floriacensis': 0.4,
							'Ioannis-Sancti-Arnulfi': 0.17}

		fig = plt.figure(figsize=(4.7,3.2))
		ax = fig.add_subplot(111, projection='3d')
		
		x1, x2, x3 = X_bar[:,0], X_bar[:,1], X_bar[:,2]

		for index, (p1, p2, p3, a, title) in enumerate(zip(x1, x2, x3, authors, titles)):

			markersymbol = 'o'
			markersize = 30

			full_title = title.split('_')[0]
			sample_number = title.split('_')[-1]
			abbrev = title.split('_')[0].split('-')
			abbrev = '.'.join([w[:3] for w in abbrev]) + '-' + sample_number

			ax.scatter(p1, p2, p3, marker='o', color='k', s=markersize, zorder=3, alpha=customized_alpha[a])

			# shingled, block of code for Bede the Venerable comparison

			# if full_title in ['Consuetudines-Floriacensis', 'Epistolas-in-catholicas', 'Illatio-s-Benedictum-Floriacum', 'Inventio-Celsi', 'Miracula-Celsi', 'Passio-Anthimi', 'Passio-SS-Tryphonis-et-Resp', 'Prologus-Argumentum', 'Sermo-de-Celso', 'Sermo-de-festivitate-sancti-Eucharii', 'Vita-Deicoli', 'Vita-S-Reginswindis', 'Vita-sancti-Firmani-posterior', 'Vita-sancti-Martini-papae', 'Vita-sancti-Severi']:
			# 	ax.scatter(p1, p2, p3, marker='o', color='w', \
			# 		s=90, zorder=2, alpha = 0.5, edgecolors ='k', linewidths = 0.2)
			# 	ax.scatter(p1, p2, p3, marker='o', color='k', \
			# 		s=markersize, zorder=3, alpha=customized_alpha['Theodericus-Floriacensis'])
			# 	ax.text(p1+0.5, p2+0.5, p3+0.5, sample_number, color='black', fontdict={'size': 6}, zorder=1)
			# elif full_title.split('-')[0] == 'shingled':
			# 	ax.scatter(p1, p2, p3, marker='o', color='k', \
			# 		s=10, zorder=1, alpha=0.02)
			# else:
			# 	ax.scatter(p1, p2, p3, marker='o', color='k', \
			# 		s=markersize, zorder=3, alpha=customized_alpha[a])
			
		ax.set_xlabel('PC 1: {}%'.format(var_pc1))
		ax.set_ylabel('PC 2: {}%'.format(var_pc2))
		ax.set_zlabel('PC 3: {}%'.format(var_pc3))

		plt.tight_layout()
		plt.show()

		fig.savefig("figs/pca.png", dpi=300, transparent=True, format='png')

class Measure_Lexical_Diversity:
	"""
	Class that measures lexical diversity of sentences in corpus.

	Parameters
	---------
	folder_location = location of .txt files
	
	Returns
	-------
	low_to_high_ld = [s, s, s, s, ...]  # list of sents from low to high diversity
	scores = [fl, fl, fl, fl, ...] # list of sorted scores from low to high

	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def go(self):
		
		scores = []
		sents = []

		for filename in glob.glob(folder_location + "/*"):
			author = filename.split("/")[-1].split(".")[0].split("_")[0]
			title = filename.split("/")[-1].split(".")[0].split("_")[1]
			text = open(filename).read().strip()
			text = re.split('\.|\?|\!', text)

			for sent in text:
				tokenized_sent = ld.tokenize(sent)
				if len(tokenized_sent) <= 40:

					masked_sent = [w for w in tokenized_sent if w in list_of_function_words]
					ttr = ld.ttr(tokenized_sent) #lexical-diversity & type-token ratio
					scores.append(ttr)
					sents.append(sent)

		low_to_high_ld = [sent for _, sent in sorted(zip(scores, sents))]
		scores = sorted(scores)

		return low_to_high_ld, scores

class t_SNE:
	"""
	Class that performs t-SNE, a tool to visualize high-dimensional data.
	It is highly recommended to use another dimensionality reduction method 
	(e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the 
	number of dimensions to a reasonable amount (e.g. 50) if the number of 
	features is very high.
	Because it is so computationally costly, especially useful for smaller
	datasets (which is often the case in attribution).

	Parameters
	---------
	
	
	Returns
	-------
	plot visualizing t-SNE

	"""
	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def plot():
		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])

		invalid_words = []
		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='normalizer', #for some reason works better than standard-scaling
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		pca = PCA(n_components=50)
		pca_X = pca.fit_transform(X)
		tsne = TSNE(n_components=2, perplexity=25, learning_rate=0.001, n_iter=40000)
		tsne_X = tsne.fit_transform(pca_X)

		fig = plt.figure(figsize=(4.7,3.2))
		ax = fig.add_subplot(111)
		x1, x2  =  tsne_X[:,0], tsne_X[:,1]

		for index, (p1, p2, a, title) in enumerate(zip(x1, x2, authors, titles)):
			
			full_title = title.split('_')[0]
			sample_number = title.split('_')[-1]
			abbrev = title.split('_')[0].split('-')
			abbrev = '.'.join([w[:3] for w in abbrev]) + '-' + sample_number

			if a == 'Adso-Dervensis':
				ax.scatter(p1, p2, marker='o', color='k', \
					s=40, zorder=2, alpha = 0.17, edgecolors ='k', linewidths = 0.2)
			elif a == 'Ioannis-Sancti-Arnulfi':
				ax.scatter(p1, p2, marker='o', color='k', \
					s=40, zorder=2, alpha = 0.4, edgecolors ='k', linewidths = 0.2)
			else:
				ax.scatter(p1, p2, marker='o', color='k', \
					s=40, zorder=2, alpha = 0.8, edgecolors ='k', linewidths = 0.2)
				# ax.text(p1+0.5, p2+0.5, abbrev, color='black', fontdict={'size': 6}, zorder=1)

			ax.set_xlabel('t-SNE-1')
			ax.set_ylabel('t-SNE-2')

		plt.tight_layout()
		plt.show()
		
		fig.savefig("figs/t-SNE.png", dpi=300, transparent=True, format='png')

class DBS:
	"""
	Class that performs DBScan
	Code inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
	"""
	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def plot():

		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])

		invalid_words = []
		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='normalizer', #for some reason works better than standard-scaling
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		pca = PCA(n_components=2)
		X = pca.fit_transform(X)

		# #############################################################################
		# Compute DBSCAN
		# epsilon range or 𝜖 indicates predefined distance around each point, (decisive whether core, border or noise point)
		
		eps = 0.46
		min_samples = len(X)

		db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
		labels = db.labels_

		no_clusters = len(np.unique(labels)) # number of clusters
		no_noise = np.sum(np.array(labels) == -1, axis=0) # number of clusters

		print('Estimated no. of clusters: %d' % no_clusters)
		print('Estimated no. of noise points: %d' % no_noise)

		# Plot results
		# Black removed and is used for noise instead.

		# Generate scatter plot for training data
		colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
		plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
		plt.xlabel('Axis X[0]')
		plt.ylabel('Axis X[1]')

		plt.show()

		fig.savefig("figs/DBScan.png", dpi=300, transparent=True, format='png')

"""
PHASE ONE
Use contrastive corpus consisting of authors Jean de Saint-Arnoul, Adso of Montier-en-Der, Abbo of Fleury, ...
Finding optimal combination of parameters and visualizing results in scenario when 3 authors are compared
"""

folder_location = ''
# Phase_One.go(folder_location) # uncomment to run

# Plot_Lowess(results_location) # uncomment to run

"""
PHASE TWO
Apply PCA with optimal parameters
"""

PrinCompAnal.plot(folder_location) # uncomment to run

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Bodoni 72'] # Font of Revue Mabillon

"""
PHASE THREE
Applying t-SNE
"""
# t_SNE.plot() # uncomment to run

"""
PHASE FOUR
Applying DBScan
"""

# DBS.plot()

"""
PHASE FIVE
Operation for finding the 'least lexically diverse' and most verbally repetitive sentence in Theoderic's oeuvre
gave intuitive example of vectorization process (in article)
"""

# low_to_high_ld, scores = Measure_Lexical_Diversity.go(folder_location) # uncomment to run
