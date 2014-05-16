from cvxopt import matrix,spmatrix,sparse,uniform,normal,setseed
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

class StructuredOCSVM:
	""" Structured One-class SVM (a.k.a Structured Anomaly Detection).
		Written by Nico Goernitz, TU Berlin, 2014
	"""
	C = 1.0	# (scalar) the regularization constant > 0
	sobj = [] # structured object contains various functions
			  # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)
	sol = [] # (vector) solution vector (after training, of course) 


	def __init__(self, sobj, C=1.0):
		self.C = C
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

		setseed(0)
		sol = 10.0*normal(DIMS,1)
		psi = matrix(0.0, (DIMS,N)) # (dim x exm)
		old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = 0

		obj = -1
		iter = 0 
		allobjs = []

		# terminate if objective function value doesn't change much
		while iter<max_iter and (iter<3 or sum(sum(abs(np.array(psi-old_psi))))>=0.001):
			print('Starting iteration {0}.'.format(iter))
			print(sum(sum(abs(np.array(psi-old_psi)))))
			iter += 1
			old_psi = matrix(psi)
			old_sol = sol
			
			# 1. linearize
			# for the current solution compute the 
			# most likely latent variable configuration
			for i in range(N):
				(foo, latent[i], psi[:,i]) = self.sobj.argmax(sol, i)
				#if i>=0:
				#	(foo, latent[i], phi[:,i]) = self.sobj.argmax(sol,i)
				#else:
				#	phi[:,i] = self.sobj.get_joint_feature_map(i)
				#	latent[i] = self.sobj.y[i]

			# 2. solve the intermediate convex optimization problem 
			kernel = Kernel.get_kernel(psi, psi)
			svm = OCSVM(kernel, self.C)
			svm.train_dual()
			threshold = svm.get_threshold()
			#inds = svm.get_support_dual()
			#alphas = svm.get_support_dual_values()
			#sol = phi[:,inds]*alphas

			#inds = svm.get_support_dual()
			#alphas = svm.get_support_dual_values()
			sol = psi*svm.get_alphas()
			print matrix([sol.trans(), old_sol.trans()]).trans()

			# calculate objective
			slacks = [max([0.0, np.single(threshold - sol.trans()*psi[:,i]) ]) for i in xrange(N)]
			obj = 0.5*np.single(sol.trans()*sol) - np.single(threshold) + self.C*sum(slacks)
			print("Iter {0}: Values (Threshold-Slacks-Objective) = {1}-{2}-{3}".format(int(iter),np.single(threshold),np.single(sum(slacks)),np.single(obj)))
			allobjs.append(float(np.single(obj)))

		print '+++++++++'
		print threshold
		print slacks
		print obj
		print '+++++++++'

		print allobjs
		print(sum(sum(abs(np.array(psi-old_psi)))))
		self.sol = sol
		self.latent = latent
		return (sol, latent, threshold)


	def apply(self, pred_sobj):
		""" Application of the StructuredOCSVM:

			anomaly_score = max_z <sol*,\Psi(x,z)> 
			latent_state = argmax_z <sol*,\Psi(x,z)> 
		"""
		N = pred_sobj.get_num_samples()
		vals = matrix(0.0, (1,N))
		lats = matrix(0.0, (1,N))
		for i in range(N):
			(vals[i], lats[i], foo) = pred_sobj.argmax(self.sol, i)

		return (vals, lats)
