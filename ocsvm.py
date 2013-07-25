from cvxopt import matrix,spmatrix,sparse
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
from kernel import Kernel  

class Ocsvm:
	"""One-class support vector machine"""

	MSG_ERROR = -1	# (scalar) something went wrong
	MSG_OK = 0	# (scalar) everything alright

	PRECISION = 10**-5 # important: effects the threshold, support vectors and speed!

	X = [] 	# (matrix) our training data
	samples = -1 	# (scalar) amount of training data in X
	dims = -1 	# (scalar) number of dimensions in input space

	C = 1.0	# (scalar) the regularization constant > 0

	ktype = 'linear' # (string) type of kernel to use
	kparam = 1.0	# (scalar) additional parameter for the kernel

	isPrimalTrained = False	# (boolean) indicates if the oc-svm was trained in primal space
	isDualTrained = False	# (boolean) indicates if the oc-svm was trained in dual space

	alphas = []	# (vector) dual solution vector
	svs = [] # (vector) support vector indices

	threshold = 0.0	# (scalar) the optimized threshold (rho)

	def __init__(self, X, C=1.0, ktype='linear', param=1.0):
		self.X = X
		self.C = C
		self.ktype = ktype
		self.kparam = param
		(self.dims,self.samples) = X.size
		print('Creating new one-class svm with {0}x{1} (dims x samples) and C={2}.'.format(self.dims,self.samples,C))
		print('Kernel is {0} with parameter (if any) set to {1}'.format(ktype,param))


	def train_primal(self):
		"""Trains a linear one-class svm (no kernel)"""
		if self.ktype!='linear':
			print('Kernel is chosen to be not linear, anyway, the primal solver only learns in input space.')

		# init return variables
		w = np.zeros((1,1))
		return Ocsvm.MSG_OK

	def get_support_dual(self):
		return self.svs

	def train_dual(self):
		"""Trains an one-class svm in dual with kernel."""
		if (self.samples<1 & self.dims<1):
			print('Invalid training data.')
			return Ocsvm.MSG_ERROR

		# number of training examples
		N = self.samples
		C = self.C

		# generate a kernel matrix
		P = Kernel.get_kernel(self.X, self.X, self.ktype, self.kparam)
		# there is no linear part of the objective
		q = matrix(0.0, (N,1))
	
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

		# find support vectors
		self.svs = []
		for i in range(N):
			if self.alphas[i]>Ocsvm.PRECISION:
				self.svs.append(i)

		# find support vectors with alpha < C for threshold calculation
		self.threshold = 10**8
		flag = False
		for i in self.svs:
			if self.alphas[i]<(C-Ocsvm.PRECISION) and flag==False:
				(self.threshold, MSG) = self.apply_dual(self.X[:,i])
				flag=True
				break

		# no threshold set yet?
		if (flag==False):
			(thres, MSG) = self.apply_dual(self.X[:,self.svs])          
			self.threshold = matrix(max(thres))

		(thres, MSG) = self.apply_dual(self.X[:,self.svs])
		T = np.array(self.threshold)[0,0]
		cnt = 0
		for i in range(len(self.svs)):
			if thres[i,0]<(T-Ocsvm.PRECISION):
				cnt += 1
	
		print(self.alphas)
		print('Found {0} support vectors. {1} of them are outliers.'.format(len(self.svs),cnt))
		print('Threshold is {0}'.format(self.threshold))
		return Ocsvm.MSG_OK

	def get_threshold(self):
		return self.threshold

	def apply_dual(self, Y):
		"""Application of a dual trained oc-svm."""

		# number of training examples
		N = self.samples

		# check number and dims of test data
		(tdims,tN) = Y.size
		if (self.dims!=tdims | tN<1):
			print('Invalid test data')
			return 0, Ocsvm.MSG_ERROR

		if (self.isDualTrained!=True):
			print('First train, then test.')
			return 0, Ocsvm.MSG_ERROR

		# generate a kernel matrix
		P = Kernel.get_kernel(Y, self.X[:,self.svs], self.ktype, self.kparam)

		# apply trained classifier
		res = matrix([dotu(P[i,:],self.alphas[self.svs]) for i in range(tN)]) 
		return res, Ocsvm.MSG_OK
