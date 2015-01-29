from cvxopt import matrix,spmatrix,sparse,exp,uniform
import numpy as np
import math as math
from so_interface import SOInterface


class SOHMM(SOInterface):
	""" Hidden Markov Structured Object."""

	ninf = -10.0**15
	start_p = [] # (vector) start probabilities
	states = -1 # (scalar) number transition states
	hotstart_tradeoff = 0.1 # (scalar) this tradeoff is used for hotstart 
							# > 1.0: transition have more weight
							# < 1.0: emission have more weight

	def __init__(self, X, y=[], num_states=2, hotstart_tradeoff=0.1):
		SOInterface.__init__(self, X, y)	
		self.states = num_states
		self.start_p = matrix(1.0, (self.states, 1))
		#self.start_p[0] = 0.2
		self.hotstart_tradeoff = hotstart_tradeoff

	def get_hotstart_sol(self):
		sol = uniform(self.get_num_dims(), 1, a=0.1,b=+1.0)
		#sol[0] = 1.0
		#sol[1] = 0.1
		#sol[2] = 1.0
		#sol[3] = 0.1

		#sol[0:self.states*self.states] = self.hotstart_tradeoff
		print('Hotstart position uniformly random with transition tradeoff {0}.'.format(self.hotstart_tradeoff))
		return sol

	def calc_emission_matrix(self, sol, idx, augment_loss=False, augment_prior=False):
		T = len(self.X[idx][0,:])
		N = self.states
		F = self.dims

		em = matrix(0.0, (N, T));
		for t in xrange(T):
			for s in xrange(N):
				for f in xrange(F):
					em[s,t] += sol[N*N + s*F + f] * self.X[idx][f,t]

		# augment with loss 
		if (augment_loss==True):
			loss = matrix(1.0, (N, T))
			for t in xrange(T):
				loss[self.y[idx][t],t] = 0.0
			em += loss
		
		if (augment_prior==True):
			prior = matrix(-0.0, (N, T))
			#prior = matrix(-10.0/float(T), (N, T))
			#prior[:,0] = -10.0
			#prior[0,:] = 0.0
			#prior[0,:] = 1.0
			em += prior

		return em

	def get_transition_matrix(self, sol):
		N = self.states
		# transition matrix
		A = matrix(0.0, (N, N))
		for i in xrange(N):
			for j in xrange(N):
				A[i,j] = sol[i*N+j]
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
			delta[i,0] = self.start_p[i] + em[i,0]
			
		# recursion
		for t in xrange(1,T):    
			for i in xrange(N):
				(delta[i,t], psi[i,t]) = max([(delta[j,t-1] + A[j,i] + em[i,t], j) for j in xrange(N)]);
		
		states = matrix(0, (1, T))
		(prob, states[T-1])  = max([delta[i,T-1], i] for i in xrange(N));    
			
		for t in reversed(xrange(1,T)):
			states[t-1] = psi[states[t],t];
		
		psi_idx = self.get_joint_feature_map(idx, states)
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
		em = self.calc_emission_matrix(sol, idx, augment_loss=False, augment_prior=False);
		
		# store scores for each position of the sequence		
		scores[0] = self.start_p[int(y[0,0])] + em[int(y[0,0]),0]
		for t in range(1,T):
			scores[t] = A[int(y[0,t-1]),int(y[0,t])] + em[int(y[0,t]),t]

		# transform for better interpretability
		if max(abs(scores))>10.0**(-15):
			scores = exp(-abs(4.0*scores/max(abs(scores))))
		else:
			scores = matrix(0.0, (1,T))

		return (float(np.single(anom_score)), scores)

	def get_joint_feature_map(self, idx, y=[]):
		y = np.array(y)
		if (y.size==0):
			y=np.array(self.y[idx])

		(foo, T) = y.shape
		N = self.states
		F = self.dims
		jfm = matrix(0.0, (self.get_num_dims(), 1))
		
		# transition part
		for i in range(N):
			(foo, inds) = np.where([y[0,1:T]==i])
			for j in range(N):
				(foo, indsj) = np.where([y[0,inds]==j]) 
				jfm[j*N+i] = float(len(indsj))/float(1.0)

		# emission parts
		for t in range(T):
			for f in range(F):
				jfm[int(y[0,t])*F + f + N*N] += self.X[idx][f,t]
		return jfm


	def get_num_dims(self):
		return self.dims*self.states + self.states*self.states


	def evaluate(self, pred): 
		(err1, err_exm1) = self.evaluate_impl(pred, change_sign=False)
		(err2, err_exm2) = self.evaluate_impl(pred, change_sign=True)
		print err1
		print err2
		print '-----------'
		if err1['fscore']>err2['fscore']:
			return (err1, err_exm1)
		return (err2, err_exm2)


	def evaluate_impl(self, pred, change_sign=False): 
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
			#loss1 = self.calc_loss(i, pred[i])
			#loss2 = self.calc_loss(i, -pred[i]+1)
			#print('{2}: loss1={0} loss2={1}'.format(loss1, loss2, i))

			#  convert into genic and intergenic regions
			seq_true = np.uint(np.sign(self.y[i]))
			seq_pred = np.uint(np.sign(pred[i]))
			if change_sign:
				# switch states
				seq_pred = np.uint(np.sign(-pred[i]+1))
			lens = len(seq_pred[0,:])

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

			if tn+fp>0:
				specificity = float(tn) / float(tn+fp)
			else:
				specificity = 0.0

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
