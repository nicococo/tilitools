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

		For more information see:
		'Learning and Evaluation with non-i.i.d Label Noise'
		Goernitz et al., AISTATS & JMLR W&CP, 2014 
	"""
	PRECISION = 10**-3 # important: effects the threshold, support vectors and speed!

	C = 1.0	# (scalar) the regularization constant > 0
	sobj = [] # structured object contains various functions
			  # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)

	threshold = 0.0	# (scalar) the optimized threshold (R^2)
	sol = [] # (vector) solution vector (after training, of course) 


	def __init__(self, sobj, C=1.0):
		self.C = C
		self.sobj = sobj


	def train_dc(self, max_iter=50):
		""" Solve the LatentSVDD optimization problem with a  
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
		latent = matrix(0.0,(1,N))
		latent_old = matrix(-1.0,(1,N))

		sol = normal(DIMS,1)
		phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = 0

		obj = -1
		iter = 0 

		# terminate if objective function value doesn't change much
		while iter<max_iter and (iter<2 or sum(abs(latent-latent_old))!=0):
			print('Starting iteration {0}.'.format(iter))
			print(sum(abs(latent-latent_old)))
			iter += 1
			latent_old = latent

			# 1. linearize
			# for the current solution compute the 
			# most likely latent variable configuration
			for i in range(N):
				(foo, latent[i], phi[:,i]) = self.sobj.argmin(sol,i,type='quadratic')

			# 2. solve the intermediate convex optimization problem 
			kernel = Kernel.get_kernel(phi,phi)
			svdd = SVDD(kernel,self.C)
			svdd.train_dual()
			threshold = svdd.get_threshold()
			inds = svdd.get_support_dual()
			alphas = svdd.get_support_dual_values()
			sol = phi[:,inds]*alphas

		self.sol = sol
		self.latent = latent
		return (sol, latent, threshold)


	def apply(self, pred_sobj):
		""" Application of the LatentSVDD:

			anomaly_score = min_z ||c*-\Psi(x,z)||^2 
			latent_state = argmin_z ||c*-\Psi(x,z)||^2 
		"""

		N = pred_sobj.get_num_samples()

		vals = matrix(0.0, (1,N))
		lats = matrix(0.0, (1,N))
		for i in range(N):
			(vals[i], lats[i], foo) = pred_sobj.argmin(self.sol,i,type='quadratic')

		return (vals, lats)
