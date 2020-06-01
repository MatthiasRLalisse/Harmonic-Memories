import tensorflow as tf
import numpy as np
from collections import defaultdict
try: from tabulate import tabulate
except: print('WARNING: GG Analyzer will not work. Install tabulate module for Python')
import re
from ..base import data_presets, KBETaskSetting
from ..utils import readTripletsData, defaultValueDict

class Analyzer(object):
	"""Class to analyze a trained Gradient Graph model. In initialization, reads in triplets and 
		ids from the data directory. the analyzer also attempts to read in human readable entity 
		names from the a file \"name2idx.tsv\" mapping human readable names to model indices. 
		Each line of \"name2idx.tsv\" must have format [name tab index]. \n
		Initializing an Analyzer requires specifying a dataset either with a \"KBETaskSetting\" object, 
		a preset \"dataName\" value, or specification of the data directory as a last resort.\n
		arg: 	\"model\", a KBEModel object.\n
		kwargs: \"task\": a KBETaskSetting object. If a task object is not passed, this 
			defaults to None and initializes a task during the __init__ call.\n
			\"dataName\": defaults to DataUnknown. Should be one of the preset values 
			for datasets packaged with gradgraph.\n
			\"dataDirectory\": directory from which to retrieve data if neither \"task\" 
			nor \"DataName\" arguments were passed."""
	def __init__(self,
		model,
		task=None,
		dataName="DataUnknown",
		dataDirectory=None
		):
		self.model = model; dataName = model.dataName
		#self.task = task if task else KBETaskSetting(dataName=dataName, filtered=True)
		self.task = KBETaskSetting(dataName=dataName, filtered=True)
		if data_presets[dataName]: 
			dataDirectory = data_presets[dataName]; self.dataName = dataName
		else: self.dataName = dataDirectory.split('/')[-1]
		#self.data = self.get_data(dataDirectory)
		self.data = self.task.data
		try: 	#build mapping from names to ids
			self.name2idx = { line.split('\t')[0]: int(line.split('\t')[1]) for line in \
						open(dataDirectory +'/'+ 'name2idx.tsv', 'r').readlines() }
		except FileNotFoundError:
			self.name2idx = self.data['entity2idx']
		self.name2idx.update({ i:i for i in range(model.n_entity) })
		try:
			idx2name = { int(line.split('\t')[0]): line.split('\t')[1].strip() for line in \
						open(dataDirectory +'/'+ 'idx2name.tsv', 'r').readlines() }
		except FileNotFoundError:
			idx2name = { self.data['entity2idx'][key]: key for key in self.data['entity2idx'] }
		self.idx2name = defaultValueDict(idx2name); self.idx2name.set_default('name unknown')
		self.relation2idx = self.data['relation2idx']
		rel2name = { self.relation2idx[key]: key for key in self.relation2idx }
		rel2name.update({ i:i for i in range(model.n_relation) })
		try:
			self.readable_triplets = [ [names1.strip().split('/'), relation.strip(), \
					names2.strip().split('/')] for names1, relation, names2 in \
					[ line.split('\t') for line in open(dataDirectory+'/'+'human_readable/'\
					+'humanReadableTriplets.txt','r').readlines() ] ]
			self.tripletsAboutDict = defaultdict(set)
			for e1_names, rname, e2_names in self.readable_triplets:
				for e1_name in e1_names:
					self.tripletsAboutDict[e1_name].add((e1_name, rname, e2_names[0]))
				for e2_name in e2_names:
					self.tripletsAboutDict[e2_name].add((e1_names[0], rname, e2_name))
					self.tripletsAboutDict[rname].add((e1_names[0], rname, e2_names[0]))
		except FileNotFoundError: pass

	def tripletsAbout(self, e_name):
		if hasattr(self, 'tripletsAboutDict'): return self.tripletsAboutDict[e_name]
		else: print('missing human readable triplet data')
	
	def search(self, name, type='e'):
		"""search database for entities or relations based on the the input.\n
		\targ: \"name\", the name to search the database contents (entity and relation inventory) for.\n
		\tkwarg: \"type\", either \'e\' (entity) or \'r\' (entity), both strings. Defaults to \'e\'.\n
		\treturns: list of entity or relation names with substrings matching the words in \"name\"."""
		words = name.split()
		if type[0] == 'e':
			source_ = self.name2idx
		elif type[0] == 'r':
			source_ = self.relation2idx
		return [ name for name in source_ if all( re.search(re.escape(word.lower()), \
								str(name).lower()) for word in words ) ]
	
	def neighbors(self, e1_name, r_name, e2_name, direction='r', top_n=5, type_constrain=True, mahalanobis=False,\
							entity_tokens=False, return_table=False, print_table=True):
		"""query the model for the top_n nearest left or right neighbors. the metric is Euclidean distance.\n
		\targs: e1_name, r_name, e2_name (names or indices of e1, r, e2).\n
		\tkwargs: \tdirection: stringright or left entity. Defaults to \'r\'; also accepts \'l\'.\n
			\ttop_n: number of neighbors to retrieve \n
			\ttype_constrain: only consider entities that satisfy a relation's type constraints'\
								'(defaults to True)\n
			\tentity_tokens: if True, retrieves neighborhoods of estimated entity tokens. '\
				'Requires the model to have specified model.mu_h_1 and model.mu_h_2 objects. \n
		\returns: None or a table of compositional embedding neighbors and token embedding neighbors '\
						'if kwarg return_table=True (defaults to False). \
									the table is also printed to stdout."""
		InDB = self.data['filter']; 
		if type_constrain:
			candDict = self.data['candidates_'+direction]
		else: 
			candDict = defaultValueDict(); 
			candDict.set_default(list(range(self.model.n_entity)))
		try: e1_ = e1_name if type(e1_name) == int else self.name2idx[e1_name]
		except KeyError: print('could not find %s in database' % (str(e1_name),)); return
		try: r_ = r_name if type(r_name) == int else self.relation2idx[r_name]
		except KeyError: print('could not find %s in database' % (str(r_name),)); return
		try: e2_ = e2_name if type(e2_name) == int else self.name2idx[e2_name]
		except KeyError: print('could not find %s in database' % (str(e2_name),)); return
		if print_table:
			print('Querying %s: %s' % ('e2' if direction=='r' else 'e1', (e1_name, r_name, e2_name),))
		true_idx = e2_ if direction=='r' else e1_
		true_name = self.idx2name[true_idx]
		candidates = candDict[r_]
		x_obj = (self.model.e2 if direction=='r' else self.model.e1) if entity_tokens else self.model.x
		mu_obj = (self.model.mu_h_2 if direction=='r' else self.model.mu_h_1) if entity_tokens else self.model.mu_h
		ranked = self.task.rankEntities(self.model, [e1_],[r_],[e2_], direction=direction, \
							type_constrain=type_constrain, filtered=False)[0] 
		placeholder_dict = 	({self.model.e1_choice: [[e1_]*len(candidates)], self.model.r_choice: [r_], \
						self.model.e2_choice:[candidates] }) if direction == 'r' else \
					({self.model.e1_choice: [candidates], self.model.r_choice: [r_], \
						self.model.e2_choice: [[e2_]*len(candidates)] })
		x_embeddings = self.model.sess.run(x_obj, placeholder_dict )
		x_embeddings = np.squeeze(x_embeddings,axis=0)
		h_embeddings = np.squeeze(self.model.sess.run(mu_obj, placeholder_dict), axis=0)
		x, h_hat = self.model.predict(e1_, r_, e2_) if not entity_tokens \
					else self.model.predict_token(e1_,r_,e2_,direction=direction)
		x, h_hat = np.expand_dims(x, axis=0), np.expand_dims(h_hat, axis=0)
		if mahalanobis:
			Precis = self.model.sess.run(self.model.Sigma_inv)
			x_distances = np.sqrt(np.einsum('bi,ij,bj->b', x-x_embeddings, Precis, x-x_embeddings))
			hat_distances = np.sqrt(np.einsum('bi,ij,bj->b', h_hat-h_embeddings, Precis, h_hat-h_embeddings))
		else:
			x_distances = np.sqrt(np.sum(np.square(x - x_embeddings), axis=1))
			hat_distances = np.sqrt(np.sum(np.square(h_hat - h_embeddings), axis=1))
		x_ranked = [ candidates[i] for i in list(np.argsort(x_distances)) ]
		hat_ranked = [ candidates[i] for i in list(np.argsort(hat_distances)) ]
		top_x_embeddings = [ idx for idx in x_ranked if idx in self.idx2name and idx != true_idx ][:top_n]
		top_hat_embeddings = [ idx for idx in hat_ranked if idx in self.idx2name and idx != true_idx ][:top_n]
		top_x_names = [ self.idx2name[idx] for idx in top_x_embeddings ]
		top_hat_names = [ self.idx2name[idx] for idx in top_hat_embeddings ]
		ranks_in_sample_x = [ ranked.index(idx) + 1 for idx in top_x_embeddings ]
		ranks_in_sample_mu = [ ranked.index(idx) + 1 for idx in top_hat_embeddings ]
		fact_in_db_mu = [ 'X' if InDB[(idx, r_, e2_)] else '' for idx in top_hat_embeddings ] 
		fact_in_db_x = [ 'X' if InDB[(idx, r_, e2_)] else '' for idx in top_x_embeddings ] 
		true_entity_info = [ ( 0, true_name, ranked.index(true_idx)+1 if true_idx in candidates else 'TypeErr', \
					'X' if InDB[(e1_, r_, e2_)] else '' , true_name, ranked.index(true_idx)+1 \
					if true_idx in candidates else 'TypeErr', 'X' if InDB[(e1_, r_, e2_)] else '' ) ]
		tab = true_entity_info + [ [ i+1, name, ranks_in_sample_x[i], fact_in_db_x[i], top_hat_names[i], \
					ranks_in_sample_mu[i], fact_in_db_mu[i] ] for i, name in enumerate(top_x_names) ]
		table = tabulate(tab, headers=['n', 'TOP X', 'X RANK', 'X IN DB', 'TOP H', 'H RANK','H IN DB'])
		if print_table: print(table)
		if return_table: return tab
		else: return top_x_embeddings, top_hat_embeddings



