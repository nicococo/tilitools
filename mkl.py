from cvxopt import matrix,spmatrix,sparse,mul,spdiag
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
import math
from kernel import Kernel  
from ssad import SSAD

class MKLWrapper:
	"""Lp-norm Multiple Kernel Learning Wrapper for convex semi-supervised anomaly detection

		Note: 
		- p-norm mkl supported
		- only dual solution is supported.

		Written by: Nico Goernitz, TU Berlin, 2013/14
	"""

	MSG_ERROR = -1	# (scalar) something went wrong
	MSG_OK = 0	# (scalar) everything alright
	PRECISION = 10**-5 # important: effects the threshold, support vectors and speed!

	samples = -1 	# (scalar) amount of training data in X
	pnorm = 2.0 # (scalar) mixing coefficient regularizer norm
	kernels = []	# (3-tensor) (=list of cvxopt.matrix) kernel matrices
	y = []	# (vector) corresponding labels (+1,-1 and 0 for unlabeled)
	cy = [] # (vector) converted label vector (+1 for pos and unlabeled, -1 for outliers)
	dm = [] # (vector) kernel mixing coefficients
	ssad = [] # (method) 
	num_kernels = 0 # (scalar) number of kernels used

	def __init__(self, ssad, kernels, y, pnorm=1.0):
		""" Constructor """
		self.kernels = kernels
		self.y = y
		self.pnorm = pnorm
		(foo,self.samples) = y.size
		self.num_kernels = len(kernels)
		self.dm = [1.0] * self.num_kernels
		self.dm = [i/float(self.num_kernels) for i in self.dm]
		self.ssad = ssad
		self.ssad.set_train_kernel(self.combine_kernels(kernels))
		print('MKL with {0} kernels.'.format(self.num_kernels))
		
		# these vectors are used in the dual optimization algorithm
		self.cy = matrix(1.0,(1,self.samples)) # cy=+1.0 (unlabeled,pos) & cy=-1.0 (neg)
		for i in range(self.samples):
			if y[0,i]==-1:
				self.cy[0,i] = -1.0


	def combine_kernels(self,kernels):
		(dim1,dim2) = kernels[0].size
		mixed = matrix(0.0, (dim1,dim2))
		for i in range(self.num_kernels):
			mixed += self.dm[i] * kernels[i] 
		return mixed

	def train_dual(self):
		pnorm = self.pnorm
		iter = 0
		lastsol = [0.0]*self.num_kernels
		while sum([abs(lastsol[i]-self.dm[i]) for i in range(self.num_kernels)])>0.001:
			# train ssad with current kernel mixing coefficients
			self.ssad.set_train_kernel(self.combine_kernels(self.kernels))
			self.ssad.train_dual()

			# calculate new kernel mixing coefficients
			lastsol = self.dm
			alphas = self.ssad.get_alphas();
			cy = self.cy

			# linear part of the objective
			norm_w_sq_m = matrix(0.0,(self.num_kernels,1))
			for j in range(self.samples):
				for k in range(self.samples):
					foo = float(cy[k])*float(cy[j])*alphas[k]*alphas[j]
					for l in range(self.num_kernels):
						norm_w_sq_m[l] += self.dm[l]*self.dm[l]*foo*self.kernels[l][j,k]

			# solve the quadratic programm
			dm = [0.0]*self.num_kernels				
			sum_norm_w = 0.0;
			for i in range(self.num_kernels):
				sum_norm_w += math.pow(norm_w_sq_m[i],pnorm/(pnorm+1.0));
			sum_norm_w = math.pow(sum_norm_w,1.0/pnorm)

			for i in range(self.num_kernels):
				dm[i] = math.pow(norm_w_sq_m[i],1.0/(pnorm+1.0))/sum_norm_w

			print('New mixing coefficients:')
			print(dm)

			dm_norm = 0.0
			for i in range(self.num_kernels):
				dm_norm += math.pow(abs(dm[i]),pnorm)
			dm_norm = math.pow(dm_norm,1.0/pnorm)

			print(dm_norm)
			self.dm = dm
			iter+=1

		print('Num iterations = {0}.'.format(iter))
		return MKLWrapper.MSG_OK


	def get_threshold(self):
		return self.ssad.get_threshold()

	def get_support_dual(self):
		return self.ssad.get_support_dual()

	def get_mixing_coefficients(self):
		return self.dm

	def apply_dual(self, kernels):
		num = len(kernels)
		(dim1,dim2) = kernels[0].size
		mixed = self.combine_kernels(kernels)
		(res,msg) = self.ssad.apply_dual(mixed);
		return res, MKLWrapper.MSG_OK
