from cvxopt import matrix,spmatrix,sparse,normal
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
import math as math

from kernel import Kernel  
from svdd import SVDD

class LatentSVDD:
	""" Latent variable support vector data description.
		Written by Nico Goernitz, TU Berlin, 2014
	"""

	MSG_ERROR = -1	# (scalar) something went wrong
	MSG_OK = 0	# (scalar) everything alright

	PRECISION = 10**-3 # important: effects the threshold, support vectors and speed!

	C = 1.0	# (scalar) the regularization constant > 0
	sobj = [] # structured object contains various functions
			  # e.g. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)

	threshold = 0.0	# (scalar) the optimized threshold (R^2)
	latent = [] # (vector) latent state 
	sol = [] # (vector) solution vector ('w')


	def __init__(self, sobj, C=1.0):
		self.C = C
		self.sobj = sobj


	def train_ql_tl(self):
		""" Solve the LatentSVDD optimization problem with the most 
		    simple sequential convex programming approach:
			quasi-linearization + trust-level 
		"""
		N = self.sobj.get_num_samples()
		DIMS = self.sobj.get_num_dims()
		
		# latent variables
		latent = matrix(0.0,(1,N))

		# solution vector
		sol = normal(DIMS,1)
		sol_old = sol
		sol_old2 = sol_old

		phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		
		# trust level related
		trust_level = 1.0
		trust_level_succ = 1.1 # increase trust_level by this for faster convergence
		trust_level_fail = 0.5 # decrease trust_level by this for better approximation
		trust_level_frac = 0.1 

		# objective function value
		obj = -1
		obj_old = 0.0
		obj_old2 = 0.0

		# terminate if objective function value doesn't change much
		iter = 0 # current iteration
		while iter<20 and (iter<3 or sum(abs(sol-sol_old))>LatentSVDD.PRECISION):
			# progress report
			print('Starting iteration {0}.'.format(iter))
			print(sum(abs(sol-sol_old)))
			iter += 1

			# switch
			sol_old2 = sol_old
			sol_old = sol
			obj_old2 = obj_old
			obj_old = obj

			# 1. linearize
			# for the current solution compute the 
			# best latent variable configuration
			for i in range(N):
				(foo, latent[i], phi[:,i]) = self.sobj.argmin(sol,i,'quadratic')

			# 2. solve the intermediate convex optimization problem 
			# build kernel matrix
			kernel = Kernel.get_kernel(phi,phi,'linear',0.0)
			svdd = SVDD(kernel,self.C)
			svdd.train_dual()
			self.threshold = svdd.get_threshold()
			inds = svdd.get_support_dual()
			alphas = svdd.get_support_dual_values()
			sol = phi[:,inds]*alphas

			if iter<=3: 
				continue

			# subject to the current trust_level:
			# ||sol_new-sol_old||_2^2 <= trust_level
			dist = (sol-sol_old).trans()*(sol-sol_old)
			if np.single(dist)>trust_level**2:
				print('Regularize new solution {0} -> {1}.'.format(dist,trust_level**2))
				sol = sol_old - (sol-sol_old)/math.pow(np.single(dist),0.5)

			# 3. update trust region
			# calculate new approximate objective and real objective
			obj = self.threshold
			obj_approx = self.threshold
			for i in range(N):
				# approximate objective
				val_approx  = (sol-phi[:,i]).trans()*(sol-phi[:,i])
				if np.single(val_approx)>self.threshold:
					obj_approx += self.C*(val_approx-self.threshold)
				# exact objective
				(val, foo, foo2) = self.sobj.argmin(sol,i,'quadratic')
				if np.single(val)>self.threshold:
					obj += self.C*(val-self.threshold)

			# decrease with approximate objective
			delta_approx = abs(np.single(obj_old - obj_approx))
			# decrease with real objective
			delta_exact = abs(np.single(obj_old - obj))
			print('Approx={0} Exact={1}'.format(delta_approx,delta_exact))

			if (delta_exact>=(delta_approx*trust_level_frac)):
				print('Trust level EXPAND')
				trust_level *= trust_level_succ
			else:
				print('Trust level FAIL')
				trust_level *= trust_level_fail
				sol = sol_old
				sol_old = sol_old2
				obj = obj_old
				obj_old = obj_old2

		# return solution, latent variable vector and message
		self.sol = sol
		self.latent = latent
		return (sol, latent, LatentSVDD.MSG_OK)
	

	def train_ql(self):
		""" Solve the LatentSVDD optimization problem with the most 
		    simple sequential convex programming approach:
			quasi-linearization 
		"""
		N = self.sobj.get_num_samples()
		DIMS = self.sobj.get_num_dims()
		
		# latent variables
		latent = matrix(0.0,(1,N))
		latent_old = matrix(-1.0,(1,N))

		# solution vector
		sol = normal(DIMS,1)
		phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		
		# objective function value
		obj = -1

		# terminate if objective function value doesn't change much
		iter = 0 # current iteration
		while iter<20 and (iter<2 or sum(abs(latent-latent_old))>LatentSVDD.PRECISION):
			# progress report
			print('Starting iteration {0}.'.format(iter))
			print(sum(abs(latent-latent_old)))
			iter += 1
			latent_old = latent

			# 1. linearize
			# for the current solution compute the 
			# best latent variable configuration
			for i in range(N):
				(foo, latent[i], phi[:,i]) = self.sobj.argmin(sol,i,'quadratic')

			# 2. solve the intermediate convex optimization problem 
			# build kernel matrix
			kernel = Kernel.get_kernel(phi,phi,'linear',0.0)
			svdd = SVDD(kernel,self.C)
			svdd.train_dual()
			self.threshold = svdd.get_threshold()
			inds = svdd.get_support_dual()
			alphas = svdd.get_support_dual_values()
			sol = phi[:,inds]*alphas

		# return solution, latent variable vector and message
		self.sol = sol
		self.latent = latent
		return (sol, latent, LatentSVDD.MSG_OK)
	
	def get_threshold(self):
		return self.threshold


	def apply(self, pred_sobj):
		""" Application of the LatentSVDD.

		"""
		# number of training examples
		N = pred_sobj.get_num_samples()
		DIMS = pred_sobj.get_num_dims()

		vals = matrix(0.0, (1,N))
		latents = matrix(0.0, (1,N))
		for i in range(N):
			(vals[i], latents[i], foo) = pred_sobj.argmin(self.sol,i,'quadratic')

		return (vals, latents, LatentSVDD.MSG_OK)
