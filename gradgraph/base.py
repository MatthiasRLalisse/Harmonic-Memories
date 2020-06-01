import tensorflow as tf
import numpy as np
import os, re, sys
import gradgraph as gg
from .tf_ops import *
from .utils import permuteList
from collections import defaultdict
from itertools import chain

path = gg.__path__[0]
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

from .utils import readTripletsData, defaultValueDict, defaultFalseDict, permuteList
eps = .25	#weight matrices constrained to have l2 norm |W| <= lambda_ - eps

trained_models_path = os.path.join(path, 'trained_models')
data_presets = defaultFalseDict({'freebase':path+'/data/FB15K', 'wordnet':path+'/data/WN18', \
					'svo':path+'/data/SVO', 'fb237':path+'/data/FB15K-237',
					'wn18rr':path+'/data/WN18RR', \
					'yago3-10': path+'/data/YAGO3-10', 
					'wn18zeroshot': path+'/data/WN0Shot',
					'fbzeroshot': path+'/data/FB0Shot', \
					'countries': path+'/data/countries' })	
		#collection of standard datasets packaged with gradgraph
		#can be passed to KBETaskSetting as 'dataName' arguments to automate dataloading

def binary_activation(x):
    cond = tf.greater(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.ones(tf.shape(x)), tf.zeros(tf.shape(x)))
    return out

class KBEModel(object):
	def __init__(self, 
		entity_dim=50, 
		relation_dim=None,
		task=None,
		h_dim=None,
		n_entity=None,
		n_relation=None,
		normalize=True,
		max_neighbors=100,
		n_types=0,
		Graph=None,
		lambda_=None,
		model_class = None,
		model_class_name=None,
		gamma=0.,
		train_dropout=0.,
		lrate=.001,
		name=None,
		model_dir=None, 
		epoch_num=None,
		dataName='DataUnKnown',
		trip=False,
                init_embeddings=None ):	#model name allows restoring previous models
		#self.result_fields = ['MR_left', 'MR_right', 'MRR_left', 'MRR_right', 'Hits1_left', \
		#			'Hits3_left', 'Hits10_left', 'Hits1_right', 'Hits3_right',\
		#			'Hits10_right', 'Hits10', 'Hits3', 'Hits1', 'MRR', 'MR']
		#print(task, entity_dim, relation_dim)
		if relation_dim is None: relation_dim = entity_dim
		if h_dim is None: h_dim = entity_dim
		self.name = name if name else 'KBEModel%iE%iR%fL' % (entity_dim, relation_dim, \
								lambda_ if lambda_ else 0.) #; print(self.name)
		self.model_dir = trained_models_path if not model_dir else model_dir
		self.normalize = normalize
		self.max_neighbors = max_neighbors; print('max_neighbors', self.max_neighbors)
		self.model_class = model_class
		try: 
			with open(os.path.join(self.model_dir, self.name+'-hyperparams.txt'),'r') as f:
				lines = { l.split()[0]: l.split()[1] for l in f.readlines() }
			print( 'restoring hyperparams from save file')
			self.model_class = lines['model_class'] if 'model_class' in lines \
						else self.model_class
			self.dataName = lines['dataName']
			self.entity_dim = int(lines['entity_dim'])
			self.relation_dim = int(lines['relation_dim'])
			self.h_dim = int(lines['h_dim'])
			self.lambda_ = None if lines['lambda']=='None' \
							else float(lines['lambda'])
			self.n_entity = int(lines['n_entity'])
			self.n_relation = int(lines['n_relation'])
			self.gamma = float(lines['gamma'])
			if self.gamma: self.n_types = int(lines['n_types'])
			self.train_dropout = float(lines['train_dropout'])\
						if 'train_dropout' in lines else 0.0
			self.max_neighbors = int(lines['max_neighbors']) if 'max_neighbors' in lines \
						else self.max_neighbors
			self.normalize = bool(lines['normalize']) if 'normalize' in lines \
						else self.normalize
		except FileNotFoundError:
			assert entity_dim and relation_dim and h_dim, \
					'must pass name of a saved model with saved hyperparameters, '\
					'or supply values for entity_dim, relation_dim, and h_dim hyperparams'
			self.entity_dim = entity_dim; self.relation_dim = relation_dim
			self.h_dim=h_dim; self.lambda_ = lambda_; self.gamma = gamma	
			self.train_dropout = train_dropout
			if task:
				self.dataName = task.dataName
				self.n_entity = task.n_entity; self.n_relation = task.n_relation
				if task.typed: self.n_types = task.n_types
				self.Graph = task.Graph
			else: 
				assert n_entity and n_relation, \
					'ERROR must pass kwarg task or kwargs n_entity and n_relation'
				self.n_entity = n_entity; self.n_relation = n_relation
				self.dataName = dataName
				if self.gamma:
					assert n_types, 'ERROR must pass n_types if gamma is not 0'
		self.lrate = lrate; self.gamma=gamma
		self.hyperparams = { 'dataName': self.dataName, 'entity_dim': self.entity_dim, \
						'relation_dim': self.relation_dim, 'h_dim': self.h_dim, \
						'lambda': self.lambda_, 'n_entity':self.n_entity,\
						'n_relation': self.n_relation, 'gamma': self.gamma,
						'train_dropout': self.train_dropout, \
						'max_neighbors': self.max_neighbors, \
						'learning_rate': self.lrate, \
						'model_class': self.model_class, \
						'normalize': self.normalize }
		self.build_embeddings()
		self.build_Mem_embeddings()
		self.build_probes()
		self.batch_dim = tf.to_int32(tf.shape(self.r)[0])
		self.mem_dim = tf.to_int32(tf.shape(self.binding_indices)[-2])
		self.M = self.build_Mem()
		self.W, self.b = self.build_params()
		self.mu_M = self.build_mu_M()
		self.e_out = self.unbind_op(self.mu_M, self.r, self.e_probe)
		try: 
			self.mu_h_1, self.mu_h_2 = self.mu_entities()
		except NotImplementedError: pass
		if gamma: 
			assert self.n_types, 'specify the number of types (perhaps by inputting a typed task)'
			self.hyperparams['n_types'] = self.n_types
			try:
				self.class_probs = self.build_classifier(trip)
			except NotImplementedError: 
				print('token embeddings not defined for this model--'\
							'cannot build classification loss')
				raise NotImplementedError()
		self.H = self.build_H()
		self.scores = self.build_scores()
		self.loss = self.build_loss()
		self.train = self.build_trainer()
		self.sess = tf.Session()
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(init)
		self.restore(epoch_num)

	def build_embeddings(self):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.e_embeddings_ = tf.get_variable("entityEmbeddings", 
				shape=[self.n_entity, self.entity_dim],
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.r_embeddings_ = tf.get_variable("relationEmbeddings", 
				shape=[self.n_relation*2, self.relation_dim], initializer=
				tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			#self.e_embeddings_ = tf.Variable(np.random.randn(self.n_entity,self.entity_dim)/np.sqrt(self.entity_dim),dtype=tf.float32)
			#self.r_embeddings_ = tf.Variable(np.random.randn(self.n_relation*2,self.entity_dim)/np.sqrt(self.relation_dim),dtype=tf.float32)
			self.e_embeddings = tf.concat([self.e_embeddings_, tf.zeros([1,self.entity_dim])], axis=0)
			self.r_embeddings = tf.concat([self.r_embeddings_, tf.zeros([1,self.relation_dim])], axis=0)
		self.e1_choice = tf.placeholder(tf.int64, shape=[None, None])
		self.r_choice = tf.placeholder(tf.int64, shape=[None])
		self.e2_choice = tf.placeholder(tf.int64, shape=[None, None])
		self.binding_indices = tf.placeholder(tf.int64, shape=[None,None,2])
		self.probe_left = tf.placeholder_with_default(False, shape=None)
		#self.binarize = tf.placeholder_with_default(False, shape=None)
		embed_transform = (lambda x: tf.nn.l2_normalize(x, axis=-1)) if self.normalize else (lambda x: x)
		self.e1_embed =  embed_transform( \
			tf.nn.embedding_lookup(self.e_embeddings, self.e1_choice))
		self.r_embed = embed_transform( \
			tf.nn.embedding_lookup(self.r_embeddings, self.r_choice))
		self.e2_embed = embed_transform( \
			tf.nn.embedding_lookup(self.e_embeddings, self.e2_choice))
		self.r = self.r_embed
	
	def build_probes(self):
		self.e_target = tf.cond(self.probe_left, true_fn=lambda:self.e1_embed, \
							false_fn=lambda:self.e2_embed)
		self.e_probe = tf.squeeze(tf.cond(self.probe_left, \
							true_fn=lambda:self.e2_embed, \
							false_fn=lambda: self.e1_embed), axis=1) 
			#NORMALIZE?
		self.e_mem_indices = tf.squeeze(tf.cond(self.probe_left, true_fn=lambda:self.e2_choice, \
							false_fn=lambda:self.e1_choice), axis=1)
	
	def mu_entities(self):
		raise NotImplementedError()
	
	def bind_op(self, r, e):
		raise NotImplementedError() 
	
	def unbind_op(self, Mem_h, r, e):
		#do any necessary memory tensor-reshaping here
		raise NotImplementedError()
	
	def build_Mem_embeddings(self):
		#self.e1_mem_embed = tf.nn.embedding_lookup(self.e_embeddings, self.binding_indices[:,:,0])
		self.r_mem_indices, self.e_mem_indices = self.binding_indices[:,:,0], self.binding_indices[:,:,1]
		self.r_mem_embed = tf.nn.embedding_lookup(self.r_embeddings, self.r_mem_indices)
		self.e_mem_embed = tf.nn.embedding_lookup(self.e_embeddings, self.e_mem_indices)
		#if self.train_dropout != 0:
		#	self.r_mem_embed = tf.nn.dropout(self.r_mem_embed, rate=self.train_dropout)	#	self.b_metric = tf.nn.dropout(self.b_metric, keep_prob=1.-self.drop_prob)
		#	self.e_mem_embed = tf.nn.dropout(self.e_mem_embed, rate=self.train_dropout)	#	self.b_metric =
	
	def build_Mem(self):
		self.W_relations = tf.get_variable("W_relations", 
				shape=[self.relation_dim+self.entity_dim, \
				self.relation_dim+self.entity_dim], initializer=
				tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.b_relations = tf.get_variable("b_relations", \
			shape=[self.n_relation, self.relation_dim+self.entity_dim])#; print(self.b_relations.shape)
		r_Mats = tf.einsum('bi,ij->bj', tf.concat([self.r, self.e_probe],axis=-1), self.W_relations)
		r_bias = tf.nn.embedding_lookup(self.b_relations, self.r_choice)
		r_scores = tf.einsum('bi,bmi->bm', r_Mats, tf.concat([self.r_mem_embed, self.e_mem_embed], axis=-1))
		#Mem_bindings = self.bind_op(self.r_mem_embed, self.e_mem_embed)
		#r_scores = tf.einsum('bmi,bi->bm', self.r, r_bias)
		zero_mask = tf.cast(tf.not_equal(self.r_mem_indices, self.n_relation*2), tf.float32)
		#print('maskshape', zero_mask.shape)
		#weights = tf.nn.sigmoid(r_scores) #tf.expand_dims( \
					#tf.not_equal(self.r_choice, self.n_relation), \
					#zero_mask, \
					#tf.float32), tf.nn.sigmoid(r_scores))
		#print('weights', weights.shape, 'scores', r_scores.shape, 'r')
		weights = tf.nn.sigmoid(r_scores)
		self.top_k_values, self.indices_ = tf.nn.top_k(weights, k=self.max_neighbors)
		#print('indices', self.indices_.shape)
		bb, mm, rr = tf.meshgrid(tf.range(self.batch_dim), \
						self.indices_, tf.range(self.relation_dim), indexing='ij')
		_, _, ee = tf.meshgrid(tf.range(self.batch_dim), \
						tf.range(self.max_neighbors), \
						tf.range(self.entity_dim), indexing='ij')
		for o in [bb,mm,rr,_,ee]: print('o1',o.shape)
		r_mem_top, e_mem_top, zero_mask_ = tf.batch_gather(self.r_mem_embed, self.indices_),\
					tf.batch_gather(self.e_mem_embed, self.indices_),\
					tf.batch_gather(zero_mask, self.indices_)
		weights_sigmoid = tf.expand_dims(tf.multiply(zero_mask_, self.top_k_values), -1) 
		self.weights_ = weights_sigmoid 
		Mem_bindings = tf.multiply(self.weights_, self.bind_op(r_mem_top, e_mem_top))
		Mem = tf.reduce_sum( Mem_bindings, axis=1 )
		if self.train_dropout != 0:
			Mem = tf.nn.dropout(Mem, keep_prob=1.-self.train_dropout)
		return Mem
	
	def build_params(self):	
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.W_ = tf.get_variable("W_", shape=[self.h_dim, self.h_dim], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			#self.W__ = tf.
			W = tf.matrix_band_part(self.W_, 0, -1)
			if self.train_dropout != 0:
				self.drop_prob = tf.placeholder_with_default(0.0, shape=())
				W = tf.nn.dropout(W, keep_prob=(1.-self.drop_prob))
			else: W = W
			W = (tf.transpose(W) + W)/2	
			self.W_filter = tf.get_variable("W_filter", shape=[self.h_dim, self.h_dim], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.b_filter = tf.get_variable("b_filter", shape=[self.h_dim], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.M_filter = tf.einsum('ij,bj->bi', self.W_filter, self.M) \
					    + tf.expand_dims(self.b_filter, 0)
			print('filter', self.M_filter.shape)
			print('m dim', self.M.shape); print(self.h_dim)
			self.Filter = tf.einsum('bi,bj->bij', self.M_filter, \
							self.M_filter) #outer product of filter embeddings
			self.W_metric = tf.multiply(tf.expand_dims(W,0), self.Filter) 	
			#self.W_metric__ = tf.matrix_band_part(self.W_metric_, 0,-1)
			#if self.train_dropout != 0:
			#	self.W_metric__ = tf.nn.dropout(self.W_metric__, keep_prob=1.-self.drop_prob)
			#else: self.W_metric = self.W_metric
			#self.W_metric = (tf.transpose(self.W_metric__)+self.W_metric__)/2.
			if self.lambda_:
				self.W_metric = tf.clip_by_norm(self.W_metric, self.lambda_-eps, axes=[-2,-1])
			else:
				self.W_metric = tf.clip_by_norm(self.W_metric, 2.5, axes=[-2,-1])
			b = tf.get_variable("b_", shape=[self.h_dim, self.entity_dim], \
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			if self.train_dropout != 0:
				b = tf.nn.dropout(b, keep_prob=1.-self.drop_prob)
			self.b_metric = tf.einsum('hi,bi->bh', b, self.e_probe)
		return W, b
	
	def build_mu_M(self):
		if not self.lambda_:
			return self.M
		else:
			b_ = self.b_metric
			self.lambdaI = self.lambda_*tf.eye(self.h_dim,dtype=tf.float32)
			self.Sigma_inv = tf.expand_dims(self.lambdaI,0) - self.W_metric
			self.Sigma = tf.matrix_inverse(self.Sigma_inv)
			print('metric tensor', self.W_metric.shape)
			print(self.M.shape); print('b_', b_.shape)
			self.m_M = 2*self.lambda_*self.M + b_
			print('m_M', self.m_M.shape);print('sigma',self.Sigma.shape)
			self.l_M = (1./4.)*tf.einsum('bi,bij,bj->b', self.m_M,self.Sigma,self.m_M)\
					-self.lambda_*tf.einsum('bi,bi->b', self.M, self.M)
			return tf.einsum('bi,bij->bj',self.m_M,self.Sigma)/2.
	
	def build_H(self):
		if self.lambda_:
			H = (1./2.)*self.l_M
		else:	
			#print(self.M.shape, self.W_metric.shape, self.b.shape)
			#H = tf.einsum('bi, bi -> b', self.M, self.M)
			H = 1./2.*(tf.einsum('bi,bij,bj->b',self.M,self.W_metric,self.M) + \
							tf.einsum('bi,bi->b', self.M,self.b_metric))
			#print(H.shape)
			#print('H checks'); print(H.shape)
			#H = tf.einsum('bi,bi->b', self.M,self.b_metric)	#if lambda_ is None, mu_h = x
			print(H.shape)
		return H
	
	def score(self):
		print('out',self.e_out.shape); print('targ', self.e_target.shape)
		scores_ = tf.expand_dims(self.e_out,1) - self.e_target
		scores = tf.reduce_sum(-tf.square(scores_), axis=-1)
		return scores
	
	def build_scores(self):	#by default, scores are Harmony values
		scores = self.score()
		self.softmax_scores = tf.nn.softmax(scores,axis=1)
		#print('scores',self.softmax_scores.shape)
		self.test_scores = tf.squeeze(self.softmax_scores,0)
		self.true_posterior = self.softmax_scores[:,0]
		return scores

	def build_loss(self):	#default loss is log-softmax of the true triplet
		self.rank_loss = -tf.nn.log_softmax(self.scores,axis=1)[:,0]
		if self.gamma:
			self.true_class_indices = tf.stack([tf.range(tf.shape(self.e1type_choice)[0]), \
								self.e1type_choice], axis=1)
			self.class_loss = tf.reduce_sum(-tf.gather_nd(self.log_class_probs, \
								self.true_class_indices), axis=0)
			return (1-self.gamma)*self.rank_loss + tf.cond(self.train_classes, \
					true_fn=lambda: self.gamma*self.class_loss, false_fn=lambda: 0.)
		else:
			return self.rank_loss
	
	def build_trainer(self):
		self.optimizer = tf.train.AdamOptimizer(self.lrate)
		return self.optimizer.minimize(self.loss)
	
	@property
	def params(self):
		W, b, e_embed, r_embed = self.sess.run([self.W, self.b, \
						self.e_embeddings, self.r_embeddings])
		params = { 'weights': W, 'bias': b, 'entity_embeddings': e_embed, \
							'relation_embeddings': r_embed }
		if self.gamma: 
			params['W_classifier'] = self.W_class
			params['b_classifier'] = self.b_class
		return { 'weights': W, 'bias': b, 'entity_embeddings': e_embed, \
							'relation_embeddings': r_embed }
	
	def build_classifier(self, trip=True):
		self.e1type_choice = tf.placeholder_with_default(tf.constant([0],dtype=tf.int32), shape=[None])
		self.train_classes = tf.placeholder_with_default(False, shape=())
		self.W_class = tf.get_variable('W_class', shape=[self.n_types, self.h_dim], \
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.b_class = tf.get_variable("b_class", shape=[self.n_types], \
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		if trip: true_mu_h = self.mu_h[:,0,:]
		else: true_mu_h = self.mu_h_1[:,0,:]
		class_layer = tf.nn.tanh(tf.einsum('ij,bj->bi', self.W_class, true_mu_h) + 
						tf.expand_dims(self.b_class, 0)) #tf.nn.sigmoid...
		class_probs = tf.nn.softmax(class_layer, axis=1)
		self.log_class_probs = tf.nn.log_softmax(class_layer, axis=1)
		return class_probs
	
	def restore(self, epoch_num):
		self.saver = tf.train.Saver(max_to_keep=15)
		re_chk = re.compile(re.escape(self.name)+'-epoch_[0-9]+\.ckpt')
		checkpoints = [ file_ for file_ in os.listdir(self.model_dir) if re_chk.match(file_) ]
		epoch_nums = [ int(line_.split('_',1)[1].split('.',1)[0]) for line_ in checkpoints ]
		if len(epoch_nums) > 0:
			if epoch_num:
				if epoch_num in epoch_nums:
					self.epoch = epoch_num 
				else:
					self.epoch = max(epoch_nums)
					print('could not find model matching epoch_num %i\nrestoring'\
							' epoch %i instead' % (epoch_num,self.epoch))
			else:
				self.epoch = max(epoch_nums)
			model_id = '%s-epoch_%i.ckpt' % (self.name, self.epoch)
			print('restoring from epoch {0} model'.format(self.epoch) + '\t' \
							+ os.path.join(self.model_dir, model_id))
			self.saver.restore(self.sess, os.path.join(self.model_dir, model_id))
			try: 
				self.results = { (int(line.split('\t')[0].split()[1]), line.split('\t')[0].split()[2]) : \
						{ obj.split()[0]: float(obj.split()[1]) for obj in line.split('\t')[1:] } \
						for line in (open(os.path.join(self.model_dir, self.name+'-results.txt'), \
						'r')).readlines() }
			except FileNotFoundError:
				self.results = {}
		else: 
			self.epoch = 0; self.results = {}
	
	def save(self, force_overwrite=False):
		#save trained model
		model_id = self.name + '-epoch_%i.ckpt' % (self.epoch,)
		model_path = os.path.join(self.model_dir, model_id)
		if not force_overwrite and any( re.match(re.compile(model_id+'*'), filename) for \
						filename in os.listdir(self.model_dir)):
			overwrite_ = input('data for model %s already exists--overwrite?'\
					' (type \'YES\' to overwrite) ' % (model_id,))
			if overwrite_ != 'YES': 
				print('model not saved'); return
		self.saver.save(self.sess, model_path)
		result_lines = [ 'epoch '+str(epoch) + ' ' + datatype + '\t' + '\t'.join([key + ' '+ \
				str(self.results[(epoch,datatype)][key]) for key in self.results[(epoch, datatype)] ])
 				for epoch, datatype in sorted(self.results.keys()) ]
		for i in range(len(result_lines)-1): result_lines[i] += '\n'
		with open(os.path.join(self.model_dir, self.name + '-results.txt'), 'w') as f:
			f.writelines(result_lines)
		hyperparams_lines = [ key + ' ' + str(self.hyperparams[key]) + '\n' for key in self.hyperparams ]
		hyperparams_lines[-1] = hyperparams_lines[-1][:-1]
		hpfile = os.path.join(self.model_dir, self.name+'-hyperparams.txt')
		with open(hpfile, 'w') as f:
			f.writelines(hyperparams_lines)
		print('saved at %s' % (model_path,)); return

	def predict(self, e1_, r_, e2_):
		x, mu_h = self.sess.run( [self.x, self.mu_h], { self.e1_choice:[[e1_]], \
						self.r_choice:[r_], self.e2_choice:[[e2_]]})
		return np.squeeze(np.squeeze(x, axis=0),axis=0), np.squeeze(np.squeeze(mu_h, axis=0),axis=0)

	def predict_token(self, e1_, r_, e2_, direction='r'):
		target, type_embed = ('mu_h_2', self.e2) if direction=='r' else ('mu_h_1', self.e1)
		if not hasattr(self, target):
			raise NotImplementedError
		e_, mu_h_ = self.sess.run( [type_embed, getattr(self, target)], {self.e1_choice:[[e1_]], \
							self.r_choice:[r_], self.e2_choice:[[e2_]]})
		return np.squeeze(np.squeeze(e_, axis=1),axis=0), \
					np.squeeze(np.squeeze(mu_h_,axis=1), axis=0)
		
class KBETaskSetting(object):
	def __init__(self,
		dataName = None,
		dataDirectory='./data/',
		typed=False,
		negsamples=100,
		batch_size=100,
		max_neighbors=500,
		filtered=True,		#if False, use raw eval setting
		type_constrain=True):	#if True, candidate entities must satisfy relational type constraints
		self.negsamples = negsamples
		self.batch_size = batch_size
		self.max_neighbors = max_neighbors
		self.typed = typed; self.filtered = filtered; self.type_constrain = type_constrain
		if data_presets[dataName]: 
			dataDirectory = data_presets[dataName]; self.dataName = dataName
			self.dataDirectory = dataDirectory
		else: self.dataName = dataDirectory.split('/')[-1]
		self.data = self.get_data(dataDirectory, typed)
		self.n_entity = max(self.data['entity2idx'].values())+1
		self.n_relation = max(self.data['relation2idx'].values())+1
		if not self.filtered: self.data['filter'] = defaultFalseDict()
		if not self.type_constrain: 
			constraints = defaultValueDict(); constraints.set_default(list(range(self.n_entity)))
			self.data['candidates_l'] = self.data['candidates_r'] = constraints
		if typed: self.n_types = max(self.data['type2idx'].values())+1
		self.Graph = self.build_Graph()
		self.FullGraph = self.build_Graph(include_valid=True)
	
	def get_data(self, dataDirectory, typed):
		return readTripletsData(dataDirectory, typed)
	
	def build_Graph(self, include_valid=False):
		Graph_ = defaultdict(list)
		D = self.data['train'].copy()
		if include_valid: 
			D = [ d + self.data['valid'][i] for i, d in enumerate(D) ]
		for i in range(len(D[0])):
			e1_idx, r_idx, e2_idx = D[0][i], D[1][i], D[2][i]
			Graph_[e1_idx].append([ r_idx+self.n_relation, e2_idx ]) 	#right entities have relation
			Graph_[e2_idx].append([ r_idx, e1_idx ])			#indices in range [n_relation, 
		return Graph_ 								#n_relation*2]
	
	def trainLoop(self, model, e1choice, rchoice, e2choice, e1choice_neg, e2choice_neg, sess=None, e1types=None):
		if not sess: sess = model.sess
		batch_size = len(rchoice)
		e1_choice_ = [ [ e1choice[j] ] for j in range(batch_size) ]
		e1_choice_neg = [ [ e1choice[j] ] + e1choice_neg[j] for j in range(batch_size) ]
		rchoice_left = rchoice 
		rchoice_right = [ r_idx+self.n_relation for r_idx in rchoice ]
		e2_choice_ = [ [ e2choice[j] ] for j in range(batch_size) ]
		e2_choice_neg = [ [ e2choice[j] ] + e2choice_neg[j] for j in range(batch_size) ]
		max_mem_e1 = np.minimum(np.maximum(model.max_neighbors, \
				max([len(self.Graph[e]) for e in e1choice])-1), self.max_neighbors)
		max_mem_e2 = np.minimum(np.maximum(model.max_neighbors, \
				max([len(self.Graph[e]) for e in e2choice])-1), self.max_neighbors)
		e1_bind_indices, e2_bind_indices = [], []
		for e1, r_l, r_r, e2 in zip(e1choice, rchoice_left, rchoice_right, e2choice):
			e1_mem = [ trip for trip in self.Graph[e1] if trip != [r_r,e2] ]
			e1_bindings = permuteList(e1_mem)[:max_mem_e1] \
						+ [[ self.n_relation*2, \
						self.n_entity]]*np.maximum(max_mem_e1-len(e1_mem),0)
			e1_bind_indices.append(e1_bindings)
			e2_mem = [ trip for trip in self.Graph[e2] if trip != [r_l,e1] ]
			e2_bindings = permuteList(e2_mem)[:max_mem_e2] \
						+ [[ self.n_relation*2, \
						self.n_entity]]*np.maximum(max_mem_e2-len(e2_mem),0)
			e2_bind_indices.append(e2_bindings)
		left_placeholders = {	model.e1_choice: e1_choice_neg, 
					model.r_choice: rchoice_left,
					model.e2_choice: e2_choice_,
					model.probe_left: True, 
					model.binding_indices: e2_bind_indices}
		right_placeholders =  {	model.e1_choice: e1_choice_, 
					model.r_choice: rchoice_right, 
					model.e2_choice: e2_choice_neg,
					model.binding_indices: e1_bind_indices}
		if model.train_dropout != 0:
			left_placeholders[model.drop_prob]: model.train_dropout
			right_placeholders[model.drop_prob]: model.train_dropout
		if e1types: 
			left_placeholders[model.e1type_choice] = e1types
			left_placeholders[model.train_classes] = True
		#print('indices', sess.run([tf.shape(model.indices_), model.indices_], left_placeholders))
		#print('weights', sess.run(model.W_relations, left_placeholders))
		#print('values', sess.run(model.weights_, left_placeholders))
		#train left entity
		batch_loss_left, null = sess.run([model.loss, model.train], left_placeholders )
		batch_loss_left = np.sum(batch_loss_left)
		#train right entity
		batch_loss_right, null = sess.run([model.loss, model.train], right_placeholders)
		batch_loss_right = np.sum(batch_loss_right)
		return batch_loss_left + batch_loss_right

	def trainEpoch(self, model, sess=None, interactive=False):
		if not sess: sess = model.sess
		epoch = model.epoch + 1
		e1s_train, rs_train, e2s_train = self.data['train'][:3]
		if self.typed: e1types_train = self.data['train'][3]
		batches_ = int(len(e1s_train)/self.batch_size)
		perm_ = np.random.permutation(len(e2s_train))
		e1s_train_p, rs_train_p, e2s_train_p = [ permuteList(l, perm_) for l in \
							[e1s_train, rs_train, e2s_train] ]
		if self.typed: e1types_train_p = permuteList(e1types_train, perm_)
		epoch_error = 0
		for i in range(batches_):
			if self.typed:
				e1choice, rchoice, e2choice, e1types = [ l[i*self.batch_size:i*self.batch_size + \
					self.batch_size] for l in [e1s_train_p, rs_train_p, \
					e2s_train_p, e1types_train_p ] ]
			else:
				e1choice, rchoice, e2choice = [ l[i*self.batch_size:i*self.batch_size + \
					self.batch_size] for l in [e1s_train_p, rs_train_p, \
					e2s_train_p ] ]
				e1types = None
			e1choice_neg = [ [ np.random.randint(self.n_entity) for n in range(self.negsamples) ] \
									for m in range(len(e1choice)) ]
			e2choice_neg = [ [ np.random.randint(self.n_entity) for n in range(self.negsamples) ] \
									for m in range(len(e2choice)) ]
			batch_loss = self.trainLoop(model, e1choice, rchoice, e2choice, \
							e1choice_neg, e2choice_neg, sess=sess, e1types=e1types)
			epoch_error += batch_loss
			if interactive: 
				sys.stdout.flush(); 
				sys.stdout.write(('\rtraining epoch %i \tbatch %i of %i \tbatch loss = %f\t\t'\
								% (epoch, i+1, batches_, batch_loss))+'\r')
		model.epoch += 1
		return epoch_error
	
	def rankEntities(self, model, entity_1s,relations_,entity_2s, direction='r', \
					 sess=None, type_constrain=None, filtered=None, full_graph=False):
		if not sess: sess = model.sess; 
		if type_constrain or type_constrain==None:
			candDict = self.data['candidates_'+direction]
		else: 
			candDict = defaultValueDict(); 
			candDict.set_default(list(range(model.n_entity)))
		filtered = filtered if filtered != None else self.filtered
		Filter = self.data['filter'] if filtered else defaultFalseDict()
		true_triplets = (entity_1s,relations_,entity_2s)
		candidates_ = []
		entities_ = []; relations__ = []; bind_indices_ = []
		Graph = self.FullGraph if full_graph else self.Graph
		for j in range(len(entity_1s)):
			entity_1, relation_, entity_2 = entity_1s[j], relations_[j], entity_2s[j]
			if direction == 'r':	
				max_mem = np.minimum( np.maximum(model.max_neighbors, \
					max([len(Graph[e]) for e in entity_1s])), self.max_neighbors)
				bind_indices = permuteList(Graph[entity_1])[:max_mem] + [[ \
						self.n_relation*2, self.n_entity]]*np.maximum(max_mem-\
						len(Graph[entity_1]),0)
				candidates = [ [entity_2] + [ e_ for e_ in candDict[relation_] \
					if e_ != entity_2 and not(Filter[(entity_1,relation_,e_)]) ] ]
				entities_ = [[entity_1]] 
			else:
				max_mem = np.minimum( np.maximum(model.max_neighbors, \
					max([len(Graph[e]) for e in entity_2s])), self.max_neighbors)
				bind_indices = permuteList(Graph[entity_2])[:max_mem] + [[ \
						self.n_relation*2, self.n_entity]]*np.maximum(max_mem-\
						len(Graph[entity_2]),0)
				candidates = [[entity_1] + [ e_ for e_ in candDict[relation_] \
					if e_ != entity_1 and not(Filter[(e_,relation_,entity_2)]) ] ]
				entities_ = [[entity_2]] 
			bind_indices_.append(bind_indices) 
			candidates_ += candidates
		if direction=='r':
			relations_right = [ r_idx+self.n_relation for r_idx in relations_ ]
			scores = [sess.run( model.test_scores, {model.e1_choice: entities_,
								model.r_choice: relations_right,
								model.e2_choice: candidates_,
								model.binding_indices: bind_indices_ })]
		else:
			relations_left = relations_
			scores = [sess.run( model.test_scores, {model.e1_choice: candidates_,
								model.r_choice: relations_left,
								model.e2_choice: entities_, 
								model.binding_indices: bind_indices_, 
								model.probe_left: True })]
		#print(scores); print(type(scores))
		candidates_perms = [ sorted( range(len(candidates)), key=lambda x:scores[j][x] )[::-1] \
							for j,candidates in enumerate(candidates_) ]
		ranked = [ [ candidates[i] for i in candidates_perms[j] ] for j,candidates in enumerate(candidates_) ]
		return ranked
	
	def rank(self, model, entity_1, relation_, entity_2, sess=None, direction='r', full_graph=False):
		if not sess: sess = model.sess
		ranked_entities = self.rankEntities(model, entity_1, relation_, entity_2, \
							sess=sess, direction=direction, full_graph=full_graph)
		if direction=='r':
			rank = [ ranks_.index(entity_2[j])+1 for j, ranks_ in enumerate(ranked_entities) ]
		else:
			rank = [ ranks_.index(entity_1[j])+1 for j,ranks_ in enumerate(ranked_entities) ]
		return rank
	
	def eval(self, model, sess=None, test_set=False, interactive=False, num_to_test=0, 
                                         full_graph=False, data_key=None, entities_to_eval=None):
		if not sess: sess = model.sess
		entities_to_eval = set(range(self.n_entity)) if entities_to_eval is None else set(entities_to_eval)
		if entities_to_eval is not None: print('n entities', len(entities_to_eval))
		datatype = data_key if data_key is not None else ('test' if test_set else 'valid')
		print('testing...\n') 
		eval_data = self.data[datatype]
		e1s_test, rs_test, e2s_test = eval_data[:3]
		test_batch_size = 1
		perm_ = np.random.permutation(len(e2s_test))
		e1s_test_p, rs_test_p, e2s_test_p = [ permuteList(l, perm_) for l in [e1s_test,rs_test,e2s_test] ]
		if num_to_test: 
			e1s_test_p, rs_test_p, e2s_test_p = [ l[:num_to_test] for l in \
									[e1s_test_p, rs_test_p, e2s_test_p] ]
		test_batches_ = int(np.ceil(len(e1s_test_p)/float(test_batch_size)))
		n_examples = 0; ranks_left = []; ranks_right = []; n_left = 0; n_right = 0
		hits_1l = 0.; hits_3l = 0.; hits_10l = 0.; hits_1r = 0.; hits_3r = 0.; hits_10r = 0.
		for k in range(test_batches_):
			c = k-1
			e1_, r_, e2_ = [ l[k*test_batch_size:k*test_batch_size + test_batch_size] \
							for l in [e1s_test_p, rs_test_p, e2s_test_p ] ]
			n_examples += len(e1_)
			eval_e1 = e1_[0] in entities_to_eval
			eval_e2 = e2_[0] in entities_to_eval
			if eval_e1:
				#print('yes left\n')
				right_rank = self.rank(model, e1_,r_,e2_, sess=sess, direction='r', full_graph=full_graph)
				right_rank_arr = np.array(right_rank,dtype=np.int32)
				hits_1r += np.sum(right_rank_arr == 1)
				hits_3r += np.sum(right_rank_arr <= 3)
				hits_10r += np.sum(right_rank_arr <= 10)
				ranks_right += right_rank
				n_right += 1
			#else: print('no left\n')
			if eval_e2:
				#print('yes right\n')
				left_rank = self.rank(model, e1_,r_,e2_, sess=sess, direction='l', full_graph=full_graph)
				left_rank_arr = np.array(left_rank)
				hits_1l += np.sum(left_rank_arr == 1)
				hits_3l += np.sum(left_rank_arr <= 3)
				hits_10l += np.sum(left_rank_arr <= 10)
				ranks_left += left_rank
				n_left += 1
			#else: print('no right\n')
			try:
				mean_rank_e1 = int(np.mean(right_rank)) if eval_e1 else None
				mean_rank_e2 = int(np.mean(left_rank)) if eval_e2 else None
				ranks_left_arr = np.array(ranks_left)
				ranks_right_arr = np.array(ranks_right)
				MRR_left = np.mean(1./ranks_left_arr)
				MRR_right = np.mean(1./ranks_right_arr)
				mean_rank = np.mean(ranks_left + ranks_right)
				hits_10 = np.divide(hits_10l+hits_10r, n_left+n_right)
				hits_3 = np.divide(hits_3l+hits_3r, n_left+n_right)
				hits_1 = np.divide(hits_1l+hits_1r, n_left+n_right)
				MRR = np.mean(1./np.array(ranks_left+ranks_right))
				if interactive: 
					sys.stdout.flush()
					sys.stdout.write(('\r\tbatch %i of %i: rank(e1) = {0} \trank(e2) = {1} '\
						'MRR = %.5f, Hits@1 = %.5f, Hits@3 = %.5f, '\
						'Hits@10 = %.5f\t\t\r' % \
						(k+1, test_batches_,  \
						MRR, \
						hits_1, \
						hits_3, \
						hits_10)).format(mean_rank_e1, mean_rank_e2) )
			except: pass
		if interactive: print('\n')
		if n_left + n_right > 0:
			hits_10 = np.divide(hits_10l+hits_10r, n_left+n_right)
			hits_3 = np.divide(hits_3l+hits_3r, n_left+n_right)
			hits_1 = np.divide(hits_1l+hits_1r, n_left+n_right)
			MRR = np.mean(1./np.array(ranks_left+ranks_right))
			mean_rank_left = np.sum(ranks_left)/n_left if n_left > 0 else None
			mean_rank_right = np.sum(ranks_right)/n_right if n_right > 0 else None
			results = {	'MR_left':mean_rank_left,
				'MR_right':mean_rank_right,
				'MRR_left':MRR_left,
				'MRR_right':MRR_right,
				'Hits1_left':hits_1l/n_left if n_left > 0 else None,
				'Hits3_left':hits_3l/n_left if n_left > 0 else None,
				'Hits10_left':hits_10l/n_left if n_left > 0 else None,
				'Hits1_right':hits_1r/n_right if n_right > 0 else None,
				'Hits3_right':hits_3r/n_right if n_right > 0 else None,
				'Hits10_right':hits_10r/n_right if n_right > 0 else None,
				'Hits10': hits_10,
				'Hits3': hits_3,
				'Hits1': hits_1,
				'MRR': MRR,
				'MR': mean_rank }
			model.results[(model.epoch, datatype)] = results
			return results
		else: return None
		




