from ..base import KBETaskSetting
import numpy as np
from collections import defaultdict
import os, re, sys
from ..utils import permuteList, defaultFalseDict
from itertools import chain
	
class KBEHoldOutTask(KBETaskSetting):
	def __init__(self,
		dataName = None,
		dataDirectory='./data/',
		typed=False,
		negsamples=100,
		batch_size=100,
		max_neighbors=500,
		filtered=True,		#if False, use raw eval setting
		type_constrain=True):	#if True, candidate entities must satisfy relational type constraints
		super(KBEHoldOutTask, self).__init__(dataName=dataName,
			dataDirectory=dataDirectory, typed=typed, negsamples=negsamples,
			batch_size=batch_size, max_neighbors=max_neighbors, 
			filtered=filtered, type_constrain=type_constrain)
		with open(os.path.join(self.dataDirectory, 'heldout_entity_ids.tsv'), 'r') as f:
			self.heldout_entities = [ int(l) for l in f.readlines() ]	
		self.heldin_entities = [ i for i in range(self.n_entity) if i not in self.heldout_entities ]
		self.testGraph = self.build_Graph(include_heldout=True)
		self.fullGraph = self.build_Graph(include_heldout=True, include_valid=True)
		self.heldout = defaultFalseDict({ e: True for e in self.heldout_entities })
	
	def build_Graph(self, include_heldout=False, include_valid=False):
		Graph_ = defaultdict(list)
		D = self.data['train']
		if include_heldout:
			D = [ D[0] + [ t[0] for t in self.data['heldout_graph'] ], 
			      D[1] + [ t[1] for t in self.data['heldout_graph'] ], 
			      D[2] + [ t[2] for t in self.data['heldout_graph'] ] ]
		#if include_heldout: 
		#	D = [ d + self.data['heldout_graph'][i] for i, d in enumerate(D) ]
		if include_valid: 
			D = [ d + self.data['valid'][i] for i, d in enumerate(D) ]
		for i in range(len(D[0])):
			e1_idx, r_idx, e2_idx = D[0][i], D[1][i], D[2][i]
			Graph_[e1_idx].append([ r_idx+self.n_relation, e2_idx ]) 	#right entities have relation
			Graph_[e2_idx].append([ r_idx, e1_idx ])			#indices in range [n_relation, 
		return Graph_ 								#n_relation*2]
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
			e1choice_neg = [ list(np.random.choice(self.heldin_entities, size=self.negsamples)) \
									for m in range(len(e1choice)) ]
			e2choice_neg = [ list(np.random.choice(self.heldin_entities, size=self.negsamples)) \
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
		#train left entity
		batch_loss_left, null = sess.run([model.loss, model.train], left_placeholders )
		batch_loss_left = np.sum(batch_loss_left)
		#train right entity
		batch_loss_right, null = sess.run([model.loss, model.train], right_placeholders)
		batch_loss_right = np.sum(batch_loss_right)
		return batch_loss_left + batch_loss_right
	###needs: to exclude heldout entities from the rankings (so remove heldout entities from left or right)
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
		Graph = self.fullGraph if full_graph else self.testGraph
		for j in range(len(entity_1s)):
			entity_1, relation_, entity_2 = entity_1s[j], relations_[j], entity_2s[j]
			if direction == 'r':	
				max_mem = np.minimum( np.maximum(model.max_neighbors, \
					max([len(Graph[e]) for e in entity_1s])), self.max_neighbors)
				bind_indices = permuteList(Graph[entity_1])[:max_mem] + [[ \
						self.n_relation*2, self.n_entity]]*np.maximum(max_mem-\
						len(Graph[entity_1]),0)
				candidates = [ [entity_2] + [ e_ for e_ in candDict[relation_] \
					if e_ != entity_2 and not Filter[(entity_1,relation_,e_)] and not self.heldout[e_] ] ]
				entities_ = [[entity_1]] 
			else:
				max_mem = np.minimum( np.maximum(model.max_neighbors, \
					max([len(Graph[e]) for e in entity_2s])), self.max_neighbors)
				bind_indices = permuteList(Graph[entity_2])[:max_mem] + [[ \
						self.n_relation*2, self.n_entity]]*np.maximum(max_mem-\
						len(Graph[entity_2]),0)
				candidates = [[entity_1] + [ e_ for e_ in candDict[relation_] \
					if e_ != entity_1 and not Filter[(e_,relation_,entity_2)] and not self.heldout[e_] ] ]
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
	def eval(self, model, sess=None, test_set=False, interactive=False, num_to_test=0, 
                                         full_graph=False, data_key=None, entities_to_eval=None):
		entities_to_eval = set(range(self.n_entity)) if entities_to_eval is None else set(entities_to_eval)
		if not sess: sess = model.sess
		#datatype = 'test' if test_set else 'valid'
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
			#print(e1_, e2_)
			#print(e1_[0] in self.heldout_entities, e2_[0] in self.heldout_entities)
			#print(self.heldout[e1_[0]], self.heldout[e2_[0]])
			if self.heldout[e1_[0]] and e1_[0] in entities_to_eval:
				right_rank = self.rank(model, e1_,r_,e2_, sess=sess, direction='r', full_graph=full_graph)
				right_rank_arr = np.array(right_rank,dtype=np.int32)
				hits_1r += np.sum(right_rank_arr == 1)
				hits_3r += np.sum(right_rank_arr <= 3)
				hits_10r += np.sum(right_rank_arr <= 10)
				ranks_right += right_rank
				n_right += 1
			if self.heldout[e2_[0]] and e2_[0] in entities_to_eval:
				left_rank = self.rank(model, e1_,r_,e2_, sess=sess, direction='l', full_graph=full_graph)
				left_rank_arr = np.array(left_rank)
				hits_1l += np.sum(left_rank_arr == 1)
				hits_3l += np.sum(left_rank_arr <= 3)
				hits_10l += np.sum(left_rank_arr <= 10)
				ranks_left += left_rank
				n_left += 1
			try:
				mean_rank_e1 = int(np.mean(right_rank)) if self.heldout[e1_[0]] else None
				mean_rank_e2 = int(np.mean(left_rank)) if self.heldout[e2_[0]] else None
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
		mean_rank_left = np.sum(ranks_left)/float(len(ranks_left))
		mean_rank_right = np.sum(ranks_right)/float(len(ranks_right))
		results = {	'MR_left':mean_rank_left,
				'MR_right':mean_rank_right,
				'MRR_left':MRR_left,
				'MRR_right':MRR_right,
				'Hits1_left':hits_1l/n_left,
				'Hits3_left':hits_3l/n_left,
				'Hits10_left':hits_10l/n_left,
				'Hits1_right':hits_1r/n_right,
				'Hits3_right':hits_3r/n_right,
				'Hits10_right':hits_10r/n_right,
				'Hits10': hits_10,
				'Hits3': hits_3,
				'Hits1': hits_1,
				'MRR': MRR,
				'MR': mean_rank }
		model.results[(model.epoch, datatype)] = results
		return results
		


