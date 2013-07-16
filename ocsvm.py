from cvxopt import matrix,spmatrix,sparse
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
import pylab as pl
from kernel import Kernel  

class Ocsvm:
    """A simple one-class svm example."""
    def __init__(self):
    	print('NEU')


    def train_primal(self, X=[], C=1.0):
    	"""Trains a linear one-class svm (no kernel)"""
    	# init return variables
    	w = np.zeros((1,1))
    	return w


    def train_dual_svdd(self, X=[], C=1.0, type='linear'):
    	"""Trains an one-class svm in dual with kernel."""
    	(dims,N) = X.size

    	# generate a kernel matrix
    	P = Kernel.get_kernel(X, X, type, 1.0)
    	# this is the diagonal of the <kernel matrix
    	q = matrix([P[i,i] for i in range(N)], (N,1))
    
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
    	pl.plot(sol['x'])
    	pl.show()
        return sol['x']

    def apply_dual_svdd(self, X, Xtest, type='linear'):
    	return 'empty'




if __name__ == '__main__':
	svm = Ocsvm()
	X = matrix([0,1,2,3,1,1,7,1], (2,4), 'd')
	X = matrix(np.random.rand(200),(2,100))
	print(X.size)
	svm.train_dual_svdd(X,0.1)
	print('finished')