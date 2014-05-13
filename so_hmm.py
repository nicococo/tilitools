from cvxopt import matrix,spmatrix,sparse,exp
import numpy as np
import math as math



class SOHMM:
	""" Hidden Markov Structured Object."""

	X = [] # (list of matricies) data 
	y = [] # (list of vectors) state sequences (if present)
	pi = [] # (vector) start probabilities

	samples = -1 # (scalar) number of training data samples
	dims = -1 # (scalar) number of input dimensions
	states = -1 # (scalar) number transition states

	def __init__(self, X, y=[], num_states=2):
		self.X = X
		self.y = y
		self.states = num_states
		self.samples = len(X)
		(self.dims, foo) = X[0].size

		self.pi = matrix(1.0, (self.states, 1))

		print('#{0} training examples with #{1} features.'.format(self.samples,self.dims))

	def argmin(self, sol, idx):
		return self.argmax(sol,idx,opt_type='quadratic')


	def argmax(self, sol, idx, add_loss=False, opt_type='linear'):
		# if labels are present, then argmax will solve
		# the loss augmented programm
		T = len(self.X[idx][0,:])
		N = self.states
		F = self.dims

		# transition matrix
		A = matrix(0.0, (N, N))
		for i in xrange(N):
			for j in xrange(N):
				A[i,j] = sol[i*N+j]


		em = matrix(0.0, (N, T));
		for t in xrange(T):
			for s in xrange(N):
				for f in xrange(F):
					em[s,t] += sol[N*N + s*F] * self.X[idx][f,t]

		if (add_loss==True):
			loss = matrix(1.0, (N, T))
			for t in xrange(T):
				loss[self.y[idx][t],t] = 0.0
			em += loss

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
		#print len(y)
		return float(sum([np.single(self.y[idx][i])!=np.single(y[i]) for i in xrange(len(y))]))


	def get_scores(self, sol, idx, y=[]):
		y = np.array(y)
		if (y.size==0):
			y=np.array(self.y[idx])

		(foo, T) = y.shape
		N = self.states
		F = self.dims
		scores = matrix(0.0, (1, T))

		score = sol.trans()*self.get_joint_feature_map(idx)
		print score

		# transition matrix
		A = matrix(0.0, (N, N))
		for i in xrange(N):
			for j in xrange(N):
				A[i,j] = sol[i*N+j]


		em = matrix(0.0, (N, T));
		for t in xrange(T):
			for s in xrange(N):
				for f in xrange(F):
					em[s,t] += sol[N*N + s*F] * self.X[idx][f,t]
		
		
		scores[0] = self.pi[int(y[0,0])] + em[int(y[0,0]),0]
		for t in range(1,T):
			scores[t] = A[int(y[0,t-1]),int(y[0,t])] + em[int(y[0,t]),t]

		#scores = matrix([scores[i]*scores[i] for i in xrange(T)])
		#scores = exp(scores/score)
		scores = scores/max(abs(scores) )
		#print scores
		return scores


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


	def get_num_samples(self):
		return self.samples


	def get_num_dims(self):
		return self.dims*self.states + self.states*self.states
