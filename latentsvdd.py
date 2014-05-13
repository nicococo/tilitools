from cvxopt import matrix,spmatrix,sparse,normal
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
from cvxopt.lapack import syev
import numpy as np
import math as math

from kernel import Kernel  
from svdd import SVDD
from ocsvm import OCSVM

import pylab as pl
import matplotlib.pyplot as plt

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

	sol = [] # (vector) solution vector (after training, of course) 


	def __init__(self, sobj, C=1.0):
		self.C = C
		self.sobj = sobj



	def train_dc_pca(self, max_iter=50):
		""" Solve the LatentPCA optimization problem with a  
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
		phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		old_phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = matrix(0.0)

		obj = -1
		iter = 0 

		# terminate if objective function value doesn't change much
		while iter<max_iter and (iter<2 or sum(sum(abs(np.array(phi-old_phi))))>=0.001):
			print('Starting iteration {0}.'.format(iter))
			print(sum(sum(abs(np.array(phi-old_phi)))))
			iter += 1
			old_phi = matrix(phi)

			# 1. linearize
			# for the current solution compute the 
			# most likely latent variable configuration
			mean = matrix(0.0, (DIMS, 1))
			for i in range(N):
				(foo, latent[i], phi[:,i]) = self.sobj.argmax(sol,i)
				mean += phi[:,i]

			mphi = matrix(phi)
			mean /= float(N)
			#for i in range(N):
				#mphi[:,i] -= mean

			# 2. solve the intermediate convex optimization problem 
			A = mphi*mphi.trans()
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

		print(sum(sum(abs(np.array(phi-old_phi)))))
		self.sol = sol
		self.latent = latent
		return (sol, latent, threshold)



	def train_dc_svm(self, max_iter=50):
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
		latent = [0.0]*N

		sol = 10*normal(DIMS,1)
		phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		old_phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = 0

		obj = -1
		iter = 0 

		# terminate if objective function value doesn't change much
		while iter<max_iter and (iter<5 or sum(sum(abs(np.array(phi-old_phi))))>=0.001):
			print('Starting iteration {0}.'.format(iter))
			print(sum(sum(abs(np.array(phi-old_phi)))))
			iter += 1
			old_phi = matrix(phi)

			# 1. linearize
			# for the current solution compute the 
			# most likely latent variable configuration
			for i in range(N):
				#(foo, latent[i], phi[:,i]) = self.sobj.argmax(sol,i)
				if i<15:
					(foo, latent[i], phi[:,i]) = self.sobj.argmax(sol,i)
				else:
					phi[:,i] = self.sobj.get_joint_feature_map(i)
					latent[i] = self.sobj.y[i]

			# 2. solve the intermediate convex optimization problem 
			kernel = Kernel.get_kernel(phi,phi)
			svm = OCSVM(kernel,self.C)
			svm.train_dual()
			threshold = svm.get_threshold()
			inds = svm.get_support_dual()
			alphas = svm.get_support_dual_values()
			sol = phi[:,inds]*alphas

		print(sum(sum(abs(np.array(phi-old_phi)))))
		self.sol = sol
		self.latent = latent
		return (sol, latent, threshold)


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
		latent = [0]*N

		sol = 10.0*normal(DIMS,1)
		phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		old_phi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = 0

		obj = -1
		iter = 0 

		# terminate if objective function value doesn't change much
		while iter<max_iter and (iter<2 or sum(sum(abs(np.array(phi-old_phi))))>=0.001):
			print('Starting iteration {0}.'.format(iter))
			print(sum(sum(abs(np.array(phi-old_phi)))))
			iter += 1
			old_phi = matrix(phi)

			# 1. linearize
			# for the current solution compute the 
			# most likely latent variable configuration
			for i in range(N):
				(foo, latent[i], phi[:,i]) = self.sobj.argmin(sol,i)

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
			(vals[i], lats[i], foo) = pred_sobj.argmin(self.sol,i)
			#(vals[i], lats[i], foo) = pred_sobj.argmax(self.sol,i)

		return (vals, lats)
