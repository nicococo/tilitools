from cvxopt import matrix,spmatrix,sparse,exp,uniform,normal
import numpy as np
import math as math
from so_interface import SOInterface


class SOPGM(SOInterface):
	""" Hidden Markov based Procaryotic Gene Model.

		- State 0 = Intergenic and always start and end state
		- State 1 = Exon start
		- State 2 = Exon end
		- State 3,4,5 = Inner exon  

		8 Allowed Transitions:
			0 -> 0 (intergenic to intergenic)
			0 -> 1 (intergenic to exon start)
			1 -> 4 (exon start to inner exon 2)
			4 -> 5 (inner exon 2 to inner exon 3)
			3 -> 4 (inner exon 1 to inner exon 2)
			5 -> 3 (inner exon 3 to inner exon 1)
			5 -> 2 (inner exon 3 to exon end) 
			2 -> 0 (exon end to intergenic)
	
		written by Nico Goernitz, TU Berlin, 2014
	"""
	ninf = -10.0**15

	state_dims_map = []
	state_dims_jfm_inds = []
	state_dims_entries = 0

	start_p = 0 # start state index
	stop_p   = 0 # end state index

	states = 6 # (scalar) number transition states
	transitions = 8 # (scalar) number of allowed transitions

	hotstart_tradeoff = 0.01 # (scalar) this tradeoff is used for hotstart 
							# > 1.0: transition have more weight
							# < 1.0: emission have more weight

	def __init__(self, X, y=[], hotstart_tradeoff=0.01, state_dims_map=[]):
		SOInterface.__init__(self, X, y)	
		self.hotstart_tradeoff = hotstart_tradeoff
		print 'number of dims: '
		print self.dims
		
		if (state_dims_map==[]):
			print 'Create default state-dimension-map.'
			# generate a default state-dimension map (every state uses all avail dims)
			self.state_dims_map = []
			for s in range(self.states):
				self.state_dims_map.append(range(self.dims))
		else:
			self.state_dims_map = state_dims_map

		print 'Calculating jfm dimension indices..'
		cnt = self.transitions
		for s in range(self.states):
			foo = matrix(0, (1, self.dims))
			for d in self.state_dims_map[s]:
				foo[d] = cnt
				cnt += 1
			self.state_dims_jfm_inds.append(foo)
		self.state_dims_entries = cnt - self.transitions

		for s in range(self.states):
			print self.state_dims_jfm_inds[s]

		print self.state_dims_entries
		print self.state_dims_map
		print self.state_dims_jfm_inds

	def get_hotstart_sol(self):
		sol = uniform(self.get_num_dims(), 1, a=-1,b=+1)
		#sol[0:8] *= self.hotstart_tradeoff
		#sol[0:8] = 0.0
		#sol[1] = -100.0
		#sol[2:6] = -1.0*np.abs(sol[2:6])
		#sol[6] = -100.0
		print('Zero transition hotstart.')
		print('Hotstart position uniformly random with transition tradeoff {0}.'.format(self.hotstart_tradeoff))
		return sol


	def calc_emission_matrix(self, sol, idx, augment_loss=False, augment_prior=False):
		T = len(self.X[idx][0,:])
		N = self.states
		F = self.dims

		em = matrix(0.0, (N, T));
		for t in xrange(T):
			for s in xrange(N):
				for f in self.state_dims_map[s]:
					#print self.state_dims_jfm_inds[s][f]
					em[s,t] += sol[self.state_dims_jfm_inds[s][f]] * self.X[idx][f,t]

		# augment with loss 
		if (augment_loss==True):
			loss = matrix(1.0, (N, T))
			for t in xrange(T):
				loss[self.y[idx][t],t] = 0.0
			em += loss

		if (augment_prior==True):
			prior = matrix(-1.0, (N, T))
			#prior[:,0] = -10.0
			prior[0,:] = 0.0
			# prior[:,-1] = -10.0
			# prior[0,-1] = 0.0
			# for t in xrange(T):
			# 	if np.sum(self.X[idx][[0,1,2,3],t])>0.0:
			# 		prior[:,t] = -10.0
			# 		prior[1,t] = 0.0
			# 	if np.sum(self.X[idx][[59,60,61,62,63],t])>0.0:
			# 		prior[:,t] = -10.0
			# 		prior[2,t] = 0.0
			em += prior

		return em


	def get_transition_matrix(self, sol):
		N = self.states
		A = matrix(self.ninf, (N, N))
		A[0,0] = sol[0] # 0 -> 0 (intergenic to intergenic)
		A[0,1] = sol[1] # 0 -> 1 (intergenic to exon start)
		A[1,4] = sol[2] # 1 -> 4 (exon start to inner exon 2)
		A[4,5] = sol[3] # 4 -> 5 (inner exon 2 to inner exon 3)
		A[3,4] = sol[4] # 3 -> 4 (inner exon 1 to inner exon 2)
		A[5,3] = sol[5] # 5 -> 3 (inner exon 3 to inner exon 1)
		A[5,2] = sol[6] # 5 -> 2 (inner exon 3 to exon end) 
		A[2,0] = sol[7] # 2 -> 0 (exon end to intergenic)
		return A


	def argmax(self, sol, idx, add_loss=False, add_prior=False, opt_type='linear'):
		# if labels are present, then argmax will solve
		# the loss augmented programm
		T = len(self.X[idx][0,:])
		N = self.states
		F = self.dims

		# get transition matrix from current solution
		A = self.get_transition_matrix(sol)
		# calc emission matrix from current solution, data points and
		# augment with loss if requested
		em = self.calc_emission_matrix(sol, idx, augment_loss=add_loss, augment_prior=add_prior)

		delta = matrix(0.0, (N, T));
		psi = matrix(0, (N, T));
		
		# initialization
		for i in xrange(N):
			if i==self.start_p:
				delta[i,0] = 0.0 + em[i,0]
			else: 
				delta[i,0] = self.ninf

		# recursion
		for t in xrange(1,T-1):    
			for i in xrange(N):
				(delta[i,t], psi[i,t]) = max([(delta[j,t-1] + A[j,i] + em[i,t], j) for j in xrange(N)]);
		for i in xrange(N):
			if i==self.stop_p:
				(delta[i,T-1], psi[i,T-1]) = max([(delta[j,T-1-1] + A[j,i] + em[i,T-1], j) for j in xrange(N)]);
			else: 
				delta[i,T-1] = self.ninf

		states = matrix(0, (1, T))
		(prob, states[T-1])  = max([delta[i,T-1], i] for i in xrange(N));    
			
		for t in reversed(xrange(1,T)):
			states[t-1] = psi[states[t],t];
		
		psi_idx = self.get_joint_feature_map(idx,states)

		val = sol.trans()*psi_idx
		return (val, states, psi_idx)


	def get_jfm_norm2(self, idx, y=[]):
		y = np.array(y)
		if (y.size==0):
			y=np.array(self.y[idx])
		jfm = self.get_joint_feature_map(idx,y)
		return jfm.trans()*jfm


	def calc_loss(self, idx, y):
		return float(sum([np.uint(self.y[idx][i])!=np.uint(y[i]) for i in xrange(len(y))]))


	def get_scores(self, sol, idx, y=[]):
		y = np.array(y)
		if (y.size==0):
			y=np.array(self.y[idx])

		(foo, T) = y.shape
		N = self.states
		F = self.dims
		scores = matrix(0.0, (1, T))

		# this is the score of the complete example
		anom_score = sol.trans()*self.get_joint_feature_map(idx)

		# transition matrix
		A = self.get_transition_matrix(sol)
		# emission matrix without loss
		em = self.calc_emission_matrix(sol, idx, augment_loss=False);
		
		# store scores for each position of the sequence		
		scores[0] = 0.0 + em[int(y[0,0]),0]
		for t in range(1,T):
			scores[t] = A[int(y[0,t-1]),int(y[0,t])] + em[int(y[0,t]),t]

		# transform for better interpretability
		if max(abs(scores))>10.0**(-15.0):
			scores = exp(-abs(4.0*scores/max(abs(scores))))
		else:
			scores = matrix(0.0, (1,T))

		return (anom_score, scores)


	def get_joint_feature_map(self, idx, y=[]):
		y = np.array(y)
		if (y.size==0):
			y=np.array(self.y[idx])

		(foo, T) = y.shape
		N = self.states
		F = self.dims
		jfm = matrix(0.0, (self.get_num_dims(), 1))
		
		# transition part
		A = matrix(0.0, (N,N))
		for i in range(N):
			(foo, inds) = np.where([y[0,1:T]==i])
			for j in range(N):
				(foo, indsj) = np.where([y[0,inds]==j]) 
				A[j,i] = len(indsj)

		jfm[0] = A[0,0] # 0 -> 0 (intergenic to intergenic)
		jfm[1] = A[0,1] # 0 -> 1 (intergenic to exon start)
		jfm[2] = A[1,4] # 1 -> 4 (exon start to inner exon 2)
		jfm[3] = A[4,5] # 4 -> 5 (inner exon 2 to inner exon 3)
		jfm[4] = A[3,4] # 3 -> 4 (inner exon 1 to inner exon 2)
		jfm[5] = A[5,3] # 5 -> 3 (inner exon 3 to inner exon 1)
		jfm[6] = A[5,2] # 5 -> 2 (inner exon 3 to exon end) 
		jfm[7] = A[2,0] # 2 -> 0 (exon end to intergenic)

		# emission parts
		for t in range(T):
			state = int(y[0,t])
			for f in self.state_dims_map[state]:
				jfm[self.state_dims_jfm_inds[state][f]] += self.X[idx][f,t]
		return jfm


	def get_num_dims(self):
		return self.state_dims_entries + self.transitions

	def evaluate(self, pred): 
		""" Convert state sequences into negative and positive regions
			and check for true- and false positives (fscore, precision, etc pp). 
			Warning! This only work for 2-state problems.
		"""

		N = self.samples
		# assume 'pred' to be correspinding to 'y'
		if len(pred)!=N:
			print len(pred)
			raise Exception('Wrong number of examples!')

		all_tp = all_fp = all_tn = all_fn = 0.0

		err_exm = {}
		err_exm['fscore'] = []
		err_exm['sensitivity'] = []
		err_exm['specificity'] = []
		err_exm['precision'] = []
		for i in xrange(N):
			#  convert into genic and intergenic regions
			seq_true = np.uint(np.sign(self.y[i]))
			seq_pred = np.uint(np.sign(pred[i]))
			lens = len(seq_pred[0,:])
			
			#loss1 = float(sum([seq_true[0,j]!=seq_pred[0,j] for j in xrange(len(seq_true))]))
			#loss2 = float(sum([seq_true[0,j]!=-seq_pred[0,j]+1 for j in xrange(len(seq_true))]))
			#if loss2<loss1:
				# switch states
			#	seq_pred = -seq_pred[i]+1
			
			# error measures
			tp = fp = tn = fn = 0.0
			isPosAvail = False
			for t in xrange(lens):
				if (seq_true[0,t]==1):
					isPosAvail = True
				fp += float(seq_true[0,t]==0 and seq_pred[0,t]==1)
				fn += float(seq_true[0,t]==1 and seq_pred[0,t]==0)
				tp += float(seq_true[0,t]==1 and seq_pred[0,t]==1)
				tn += float(seq_true[0,t]==0 and seq_pred[0,t]==0)
			if tp+fp+tn+fn!=lens:
				print 'error'
			all_fn += fn
			all_tn += tn
			all_fp += fp
			all_tp += tp

			if tp+fn>0:
				sensitivity = float(tp) / float(tp+fn)
			else:
				sensitivity = 1.0
				if isPosAvail:
					sensitivity = 0.0
			
			specificity = float(tn) / float(tn+fp)

			if tp+fp>0:
				precision = float(tp) / float(tp+fp)
			else:
				precision = 1.0
				if isPosAvail:
					precision = 0.0
			if precision+sensitivity>0.0:
				fscore = 2.0*precision*sensitivity / float(precision+sensitivity)
			else:
				fscore = 0.0
			err_exm['fscore'].append(fscore)
			err_exm['sensitivity'].append(sensitivity)
			err_exm['specificity'].append(specificity)
			err_exm['precision'].append(precision)


		if all_tp+all_fn>0:
			sensitivity = float(all_tp) / float(all_tp+all_fn)
		else:
			sensitivity = 0.0
		if all_tn+all_fp>0:
			specificity = float(all_tn) / float(all_tn+fp)
		else:
			specificity = 0.0
		if all_tp+all_fp>0:
			precision = float(all_tp) / float(all_tp+all_fp)
		else:
			precision = 0.0
		err = {}
		if precision+sensitivity>0.0:
			err['fscore'] = 2.0*precision*sensitivity / float(precision+sensitivity)
		else:
			err['fscore'] = 0.0
		err['sensitivity'] = sensitivity
		err['specificity'] = specificity
		err['precision'] = precision

		return (err, err_exm)
