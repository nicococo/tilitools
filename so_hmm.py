from cvxopt import matrix,spmatrix,sparse,exp
import numpy as np
import math as math
from so_interface import SOInterface


class SOHMM(SOInterface):
	""" Hidden Markov Structured Object."""

	pi = [] # (vector) start probabilities
	states = -1 # (scalar) number transition states

	def __init__(self, X, y=[], num_states=2):
		SOInterface.__init__(self, X, y)	
		self.states = num_states
		self.pi = matrix(0.0, (self.states, 1))

	def calc_emission_matrix(self, sol, idx, augment_loss=False):
		T = len(self.X[idx][0,:])
		N = self.states
		F = self.dims

		em = matrix(0.0, (N, T));
		for t in xrange(T):
			for s in xrange(N):
				for f in xrange(F):
					em[s,t] += sol[N*N + s*F] * self.X[idx][f,t]

		# augment with loss 
		if (augment_loss==True):
			loss = matrix(1.0, (N, T))
			for t in xrange(T):
				loss[self.y[idx][t],t] = 0.0
			em += loss

		return em

	def get_transition_matrix(self, sol):
		N = self.states

		# transition matrix
		A = matrix(0.0, (N, N))
		for i in xrange(N):
			for j in xrange(N):
				A[i,j] = sol[i*N+j]

		return A


	def argmax(self, sol, idx, add_loss=False, opt_type='linear'):
		# if labels are present, then argmax will solve
		# the loss augmented programm
		T = len(self.X[idx][0,:])
		N = self.states
		F = self.dims

		# get transition matrix from current solution
		A = self.get_transition_matrix(sol)
		# calc emission matrix from current solution, data points and
		# augment with loss if requested
		em = self.calc_emission_matrix(sol, idx, augment_loss=add_loss)

		delta = matrix(0.0, (N, T));
		psi = matrix(0, (N, T));
		# initialization
		for i in xrange(N):
			delta[i,0] = self.pi[i] + em[i,0]
			
		# recursion
		for t in xrange(1,T):    
			for i in xrange(N):
				(delta[i,t], psi[i,t]) = max([(delta[j,t-1] + A[j,i] + em[i,t], j) for j in xrange(N)]);
		
		states = matrix(0, (1, T))
		(prob, states[T-1])  = max([delta[i,T-1], i] for i in xrange(N));    
			
		for t in reversed(xrange(1,T)):
			states[t-1] = psi[states[t],t];
		
		jfm = self.get_joint_feature_map(idx,states)
		val = sol.trans()*jfm
		return (val,states,jfm)
		

	def get_jfm_norm2(self, idx, y=[]):
		y = np.array(y)
		if (y.size==0):
			y=np.array(self.y[idx])
		jfm = self.get_joint_feature_map(idx,y)
		return jfm.trans()*jfm


	def calc_loss(self, idx, y):
		return float(sum([np.single(self.y[idx][i])!=np.single(y[i]) for i in xrange(len(y))]))


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
		scores[0] = self.pi[int(y[0,0])] + em[int(y[0,0]),0]
		for t in range(1,T):
			scores[t] = A[int(y[0,t-1]),int(y[0,t])] + em[int(y[0,t]),t]

		# transform for better interpretability
		scores = exp(-(scores/max(abs(scores)) +1.0) )
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
		for i in range(N):
			(foo, inds) = np.where([y[0,1:T]==i])
			#print y[0,1:T]
			#print len(inds)

			for j in range(N):
				(foo, indsj) = np.where([y[0,inds]==j]) 
				jfm[i*N+j] = len(indsj)

		# emission parts
		for t in range(T):
			for f in range(F):
				jfm[int(y[0,t])*F + f + N*N] += self.X[idx][f,t]
		
		return jfm


	def get_num_dims(self):
		return self.dims*self.states + self.states*self.states
