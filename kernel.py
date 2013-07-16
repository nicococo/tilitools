from cvxopt import matrix,spmatrix,sparse
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
import pylab as pl

class Kernel:
	"""Construct kernels."""

	def __init__(self):
		pass

	@staticmethod
	def get_kernel(X, Y, type='linear', param=1.0):
		"""Calculates a kernel given the data X and Y (dims x exms)"""
		(Xdims,Xn) = X.size
		(Ydims,Yn) = Y.size
    	
		kernel = matrix(1.0)
		if type=='linear':
			print('Creating linear kernel with size {0}x{1}.'.format(Xn,Yn))
			kernel = matrix([ dotu(X[:,i],Y[:,j]) for j in range(Yn) for i in range(Xn)], (Xn,Yn), 'd')
		
		return kernel