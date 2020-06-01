import tensorflow as tf
import numpy as np
from ..base import KBEModel

class HTPR(KBEModel):
	"""Harmonic Tensor Product Representation class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by the tensor product of e1, r, and e2, with dimensionality
	dim(e1)*dim(r)*dim(e2). 
	kwargs: n_entity=None,
		n_relation=None,
		entity_dim=5,
		task=None, 
		lambda_=None,
		lrate=.001,
		model_dir='trained_models',
		dataName='DataUnknown',
		name=None,
		epoch_num=None"""
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=5, 
		relation_dim=None,
		task=None,
		lambda_=None,
		gamma=0.0,
		train_dropout=0.,
		max_neighbors=100,
		lrate=.001,
		model_dir=None,
		dataName='DataUnknown',
		name=None,
		epoch_num=None ):
		if not relation_dim: relation_dim = entity_dim
		if task is not None: dataName = task.dataName
		if not name:
			name = 'HTPR%ieD%irD%sL-%s' % (entity_dim, relation_dim, str(lambda_) \
							if lambda_ else 'inf', dataName)
			if gamma: name += 'G%.3f' % (gamma)
			name += ('M'+ str(max_neighbors) if max_neighbors is not None 
						else (str(task.max_neighbors) if task is not None 
						else 'None') )
		h_dim = relation_dim * entity_dim
		super(HTPR, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					model_class=HTPR, 
					task=task,
					lambda_=lambda_,
					gamma=gamma,
					train_dropout=train_dropout,
					max_neighbors=max_neighbors,
					normalize=True,
					lrate=lrate,
					name=name, 
					dataName = dataName,
					model_dir=model_dir,
					epoch_num=epoch_num)
		#self.mu_h_1, self.mu_h_2 = self.mu_entities()

	def bind_op(self, r, e):
		tpr = tf.einsum('bmi,bmj->bmij', r, e)
		#print(r, r.shape); print(e, e.shape)
		#ee = tf.expand_dims(e, 3)
		#rr = tf.expand_dims(r, 2); 
		#print(rr,rr.shape);print(ee,ee.shape)
		#tpr = tf.multiply(rr, ee)
		batchdim = tf.cast(tf.shape(e)[0], tf.int32)
		memdim = tf.cast(tf.shape(e)[1], tf.int32)
		unravelled_tpr = tf.reshape(tpr, \
					[batchdim, memdim, self.h_dim])
		print(unravelled_tpr.shape, r.shape, e.shape)
		return unravelled_tpr
	
	def unbind_op(self, Mem_h, r, e):	#0 is left direction, 1 is right, 
						#meaning we extract left (right) entities 
		#do any necessary memory tensor-reshaping here
		batchdim = tf.cast(tf.shape(Mem_h)[0], tf.int32)
		self.mu_M_tpr = tf.reshape(Mem_h, [batchdim, self.relation_dim, self.entity_dim])
		#self.ee = e; self.rr = r
		#self.mu_M_tpr = tf.reshape(ravel1, [batchdim, self.relation_dim, self.entity_dim])
		#right_probe = lambda: tf.einsum('bj,bjk->bk', r, \
		#			tf.einsum('bi,bijk->bjk', e, self.mu_M_tpr))
		#print(M_tpr.shape)
		#M1 = 
		#left_probe = lambda: tf.einsum('bij,bj->bi', self.mu_M_tpr, r)
		probe = tf.einsum('bi,bij->bj', r, self.mu_M_tpr)
		#print(Mem_h.shape, e.shape, r.shape)
		return probe #tf.cond(self.probe_left, true_fn=left_probe, false_fn=right_probe)
	
	def score(self):
		scores = tf.reduce_sum(tf.multiply(tf.expand_dims(self.e_out,1),self.e_target), axis=-1)
		return scores	
	
	def mu_entities(self):
		raise NotImplementedError()



