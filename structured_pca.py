from cvxopt import matrix,spmatrix,sparse,uniform,normal,setseed
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
from cvxopt.lapack import syev
import numpy as np
import math as math

from kernel import Kernel  

class StructuredPCA:
	""" Structured Extension for Principle Component Analysis.
		Written by Nico Goernitz, TU Berlin, 2014
	"""
	sobj = [] # structured object contains various functions
			  # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)
	sol = [] # (vector) solution vector (after training, of course) 


	def __init__(self, sobj):
		self.sobj = sobj


	def train_dc(self, max_iter=50):
		""" Solve the optimization problem with a  
		    sequential convex programming/DC-programming
		    approach: 
		    Iteratively, find the most likely configuration of
		    the latent variables and then, optimize for the
		    model parameter using fixed latent states.
		"""
		N = self.sobj.get_num_samples()
		DIMS = self.sobj.get_num_dims()
		
		# intermediate solutions
		# latent variables
		latent = [0.0]*N

		sol = normal(DIMS,1)
		psi = matrix(0.0, (DIMS,N)) # (dim x exm)
		old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = matrix(0.0)

		obj = -1
		iter = 0 

		# terminate if objective function value doesn't change much
		while iter<max_iter and (iter<2 or sum(sum(abs(np.array(psi-old_psi))))>=0.001):
			print('Starting iteration {0}.'.format(iter))
			print(sum(sum(abs(np.array(psi-old_psi)))))
			iter += 1
			old_psi = matrix(psi)

			# 1. linearize
			# for the current solution compute the 
			# most likely latent variable configuration
			mean = matrix(0.0, (DIMS, 1))
			for i in range(N):
				(foo, latent[i], psi[:,i]) = self.sobj.argmax(sol, i, add_prior=True)
				mean += psi[:,i]

			mpsi = matrix(psi)
			mean /= float(N)
			#for i in range(N):
				#mphi[:,i] -= mean

			# 2. solve the intermediate convex optimization problem 
			A = mpsi*mpsi.trans()
			print A.size
			W = matrix(0.0, (DIMS,DIMS))
			syev(A,W,jobz='V')
			print W
			print A
			print A*A.trans()
			#sol = (W[3:,0].trans() * A[:,3:].trans()).trans()
			#sol = (W[3:,0].trans() * A[:,3:].trans()).trans()
			sol = A[:,DIMS-1]
			print sol

		print(sum(sum(abs(np.array(psi-old_psi)))))
		self.sol = sol
		self.latent = latent
		return (sol, latent, threshold)




	def apply(self, pred_sobj):
		""" Application of the StructuredPCA:

			score = max_z <sol*,\Psi(x,z)> 
			latent_state = argmax_z <sol*,\Psi(x,z)> 
		"""
		N = pred_sobj.get_num_samples()
		vals = matrix(0.0, (N,1))
		structs = []
		for i in range(N):
			(vals[i], struct, foo) = pred_sobj.argmax(self.sol, i, add_prior=True)
			structs.append(struct)

		return (vals, structs)
