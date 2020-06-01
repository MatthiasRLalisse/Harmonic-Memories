import tensorflow as tf
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10

def cconv(x, y):
	x_fft_ = tf.fft(tf.complex(x,0.0))
	#e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
	y_fft_ = tf.fft(tf.complex(y,0.0))
	x_fft = x_fft_ #+ tf.complex(tf.to_float(tf.equal(x_fft_, 0.)),0.)*no_zeros
	y_fft = y_fft_ #+ tf.complex(tf.to_float(tf.equal(y_fft_, 0.)),0.)*no_zeros
	return tf.cast(tf.real(tf.ifft(tf.multiply(tf.conj(x_fft),\
                                             y_fft))),dtype=tf.float32)

def ccorr(x, y):
	x_fft_ = tf.fft(tf.complex(x,0.0))
	#e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
	y_fft_ = tf.fft(tf.complex(y,0.0))
	x_fft = x_fft_ #+ tf.complex(tf.to_float(tf.equal(x_fft_, 0.)),0.)*no_zeros
	y_fft = y_fft_ #+ tf.complex(tf.to_float(tf.equal(y_fft_, 0.)),0.)*no_zeros
	return tf.cast(tf.real(tf.ifft(tf.multiply(x_fft,\
                                             y_fft))),dtype=tf.float32)



class HHolE(KBEModel):
	"""Harmonic Holographic Embedding class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by:
		r * ifft( conj(fft(e1)) * fft(e2))) 	(* is elementwise multiplication)
	which is a computationally efficient formula for the circular convolution of vectors e1, e2.
	
	kwargs: n_entity=None,
		n_relation=None,
		entity_dim=50,
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
		Graph=None,
		entity_dim=50,
		task=None, 
		lambda_=None,
		max_neighbors=100,
		lrate=.001,
		gamma=0.,
		train_dropout=0.,
		model_dir=None,
		dataName='DataUnknown',
		name=None,
		epoch_num=None,
		trip=False ):
		if not name: 
			name = 'HHolE%iD%sL%.2fdrop' % (entity_dim, str(lambda_) \
						if lambda_ else 'inf', train_dropout)	
							#sets naming convention for this model
			name += ('M'+ str(max_neighbors) if max_neighbors is not None 
						else (str(task.max_neighbors) if task is not None 
						else 'None') )
			if gamma: name += 'G%.4f' % (gamma)
			if trip: name += 'Trip'
			name += '-'+ (dataName if dataName != 'DataUnknown' else \
					(task.dataName if task is not None else 'DataUnKnown'))
		#print(name); print(task.dataName); print(dataName)
		relation_dim = h_dim = entity_dim
		super(HHolE, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					model_class='HHolE', 
					Graph=Graph,
					task=task,
					lambda_=lambda_,
					gamma=gamma,
					train_dropout=train_dropout,
					max_neighbors=max_neighbors,
					normalize=False,
					lrate=lrate,
					name=name,
					model_dir=model_dir,
					epoch_num=epoch_num,
					dataName=dataName,
					trip=trip)

	def bind_op(self, r, e):
		return cconv(r, e)
	
	def unbind_op(self, Mem_h, r, e):	#0 is left direction, 1 is right, 
						#meaning we extract left (right) entities 
		#do any necessary memory tensor-reshaping here
		#right_probe = lambda: cconv(r, ccorr(e, Mem_h))
		#left_probe = lambda: cconv(r, cconv(e, Mem_h))
		probe = ccorr(r, Mem_h)
		return probe #tf.cond(self.probe_left, true_fn=left_probe, false_fn=right_probe)
	
	def whiteningTransform(self, Embeddings, embeddings_, alpha=.2):
		#E = tf.nn.l2_normalize(Embeddings)
		E = tf.stop_gradient(Embeddings)
		mean_e = tf.expand_dims(tf.reduce_mean(Embeddings, axis=0),0)
		E_centered = E - mean_e; I = tf.eye(self.h_dim)
		E_Cov_empirical = tf.matmul(E_centered, E_centered, \
					transpose_a=True)/tf.cast(tf.shape(E)[0], tf.float32)
		E_Cov = (1-alpha)*E_Cov_empirical + alpha*I
		S, U, _ = tf.linalg.svd(E_Cov)
		E_Prec_sqrt = tf.matmul(U, tf.matmul(tf.diag(1./tf.sqrt(S)), \
								U, transpose_b=True))
        	#ZCAInverse_ = tf.matmul(U, tf.matmul(tf.diag(tf.sqrt(S+self.eps)), \
                #                                        U, transpose_b=True))
		#E_Prec = tf.linalg.inv(E_Cov, adjoint=True)
		#E_Prec_sqrt = tf.linalg.sqrtm(E_Prec)
		#self.e, self.v = tf.linalg.eigh(tf.stop_gradient(E_Cov))
		#self.e, self.v = tf.stop_gradient(self.e), self.v
		#Mis = np.dot(v, np.dot(np.diag(1./np.sqrt(e)), v.T))
		#self.E_Prec_sqrt = tf.matmul(tf.matmul(self.v, tf.diag(1./tf.sqrt(tf.maximum(self.e, 1e-6)))), self.v, transpose_b=True)
		E_stdev = tf.expand_dims(tf.sqrt(tf.reduce_mean(tf.square(E_centered), axis=0)),0)
		#self.E_Prec_sqrt = tf.diag(1./E_stdev)
		dim = tf.cast(tf.shape(E_centered)[-1], tf.float32)
		#print(embeddings_)
		if type(embeddings_) is list:
			#out = [ tf.tensordot(e - tf.reshape(mean_e, \
			#		shape=[1 for i in e.shape[2:]]  + [1,int(E.shape[1])]), \
			#		E_Prec, axes=[-1,0])/tf.sqrt(dim) for e in embeddings_ ]
			#out = [ tf.tensordot(e - tf.reshape(mean_e, \
			#		shape=[1 for i in e.shape[2:]]  + [1,int(E.shape[1])]), \
			#		self.E_Prec_sqrt, axes=[-1,0])/tf.sqrt(dim) for e in embeddings_ ]
			out = [ e/(tf.sqrt(dim)*E_stdev) for e in embeddings_ ]
		else: #out = tf.tensordot(embeddings_, E_Prec, axes=[-1,0])/tf.sqrt(dim)
			#out = embeddings_/(tf.sqrt(dim)*E_stdev)
			out = tf.tensordot(embeddings_, self.E_Prec_sqrt, axes=[-1,0])/dim
		return out
	
	def build_probes(self):
		whiten_entities = [self.e1_embed, self.e2_embed, self.e_mem_embed ]
		whiten_relations = [ self.r, self.r_mem_embed ]
		Embeds = tf.concat([self.e_embeddings_, self.r_embeddings_], axis=0)
		e1_embed, e2_embed, e_mem_embed, r, r_mem_embed = \
				self.whiteningTransform(Embeds, whiten_entities+whiten_relations)
		#e1_embed, e2_embed, e1_mem_embed, e2_mem_embed = \
		#		self.whiteningTransform(self.e_embeddings, whiten_entities)
		#r, r_mem_embed = self.whiteningTransform(self.r_embeddings, whiten_relations)
		self.e_target = tf.cond(self.probe_left, true_fn=lambda:e1_embed, \
							false_fn=lambda:e2_embed)
		self.e_probe = tf.squeeze(tf.cond(self.probe_left, true_fn=lambda:e2_embed, \
							false_fn=lambda: e1_embed),1)
		self.e_mem_indices = tf.squeeze(tf.cond(self.probe_left, true_fn=lambda:self.e2_choice, \
							false_fn=lambda:self.e1_choice), axis=1)
	
	#def score(self):
	#	scores = tf.reduce_sum(tf.multiply(tf.expand_dims(self.e_out,1),self.e_target), axis=-1)
	#	return scores	
	
	def mu_entities(self):
		raise NotImplementedError()



