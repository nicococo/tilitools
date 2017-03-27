from cvxopt import matrix,spmatrix,sparse
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
from kernel import Kernel  

class SVDD:
	"""Support vector data description"""

	MSG_ERROR = -1	# (scalar) something went wrong
	MSG_OK = 0	# (scalar) everything alright

	PRECISION = 10**-3 # important: effects the threshold, support vectors and speed!

	kernel = [] 	# (matrix) our training kernel
	norms = []
	samples = -1 	# (scalar) amount of training data in X

	C = 1.0	# (scalar) the regularization constant > 0

	isDualTrained = False	# (boolean) indicates if the oc-svm was trained in dual space
	alphas = []	# (vector) dual solution vector
	svs = [] # (vector) support vector indices
	threshold = 0.0	# (scalar) the optimized threshold (rho)

	obj_primal = 0.0 # (scalar) primal objective value
	obj_dual = 0.0  # (scalar) dual objective value

	def __init__(self, kernel, C=1.0):
		self.kernel = kernel
		self.C = C
		(self.samples,foo) = kernel.size
		self.norms = matrix([self.kernel[i,i] for i in range(self.samples)])
		print('Creating new SVDD with {0} samples and C={1}.'.format(self.samples,C))



	def train_dual(self):
		"""Trains an one-class svm in dual with kernel."""
		if (self.samples<1):
			print('Invalid training data.')
			return SVDD.MSG_ERROR

		# number of training examples
		N = self.samples
		C = self.C

		# generate a kernel matrix
		P = self.kernel

		# this is the diagonal of the <kernel matrix
		q = matrix([0.5*P[i,i] for i in range(N)], (N,1))
	
		# sum_i alpha_i = A alpha = b = 1.0
		A = matrix(1.0, (1,N))
		b = matrix(1.0, (1,1))

		# 0 <= alpha_i <= h = C
		G1 = spmatrix(1.0, range(N), range(N))
		G = sparse([G1,-G1])
		h1 = matrix(C, (N,1))
		h2 = matrix(0.0, (N,1))
		h = matrix([h1,h2])

		sol = qp(P,-q,G,h,A,b)
		
		# mark dual as solved
		self.isDualTrained = True

		# store solution
		self.alphas = sol['x']

		self.obj_primal = sol['primal objective']
		self.obj_dual = sol['dual objective']

		# find support vectors
		self.svs = []
		for i in range(N):
			if self.alphas[i]>SVDD.PRECISION:
				self.svs.append(i)

		# find support vectors with alpha < C for threshold calculation
		self.threshold = 10**8
		flag = False
		for i in self.svs:
			if self.alphas[i]<(C-SVDD.PRECISION) and flag==False:
				(self.threshold, MSG) = self.apply_dual(self.kernel[i,self.svs],self.norms[i])
				flag=True
				break

		# no threshold set yet?
		if (flag==False):
			(thres, MSG) = self.apply_dual(self.kernel[self.svs,self.svs],self.norms[self.svs])          
			self.threshold = matrix(min(thres))

		print('Threshold is {0}'.format(self.threshold))
		return SVDD.MSG_OK
	


	def set_train_kernel(self, kernel):
		(dim1,dim2) = kernel.size
		if (dim1!=dim2 and dim1!=self.samples):
			print('(Kernel) Wrong format.')
			return SVDD.MSG_ERROR
		self.kernel = kernel;
		self.norms = [self.kernel[i,i] for i in range(self.samples)]
		return SVDD.MSG_OK

	
	def get_objectives(self):
		return (self.obj_primal, self.obj_dual)

	def get_threshold(self):
		return self.threshold

	def get_alphas(self):
		return self.alphas


	def get_support_dual(self):
		return self.svs


	def get_support_dual_values(self):
		return self.alphas[self.svs]
	

	def apply_dual(self, k, norms):
		"""Application of a dual trained SVDD.
		   k \in m(test_data x train support vectors)
		   norms \in (test_data x 1)
		"""
		# number of training examples
		N = len(self.svs)
		(tN,foo) = k.size

		if (self.isDualTrained!=True):
			print('First train, then test.')
			return 0, SVDD.MSG_ERROR

		Pc = self.kernel[self.svs,self.svs]
		resc = matrix([dotu(Pc[i,:],self.alphas[self.svs]) for i in range(N)]) 
		resc = dotu(resc,self.alphas[self.svs])
		res = resc - 2*matrix([dotu(k[i,:],self.alphas[self.svs]) for i in range(tN)]) + norms
		return res, SVDD.MSG_OK
