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
	slacks = None
	svs_inds = None
	threshold = 0.0
	mean_psi = None
	norm_ord = 1

	def __init__(self, sobj, C=1.0, norm_ord=1):
		self.C = C
		self.sobj = sobj
		self.norm_ord = norm_ord

	def train_dc(self, zero_shot=False, max_iter=50, hotstart=matrix([])):
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

		#setseed(0)
		sol = self.sobj.get_hotstart_sol()
		#sol[0:4] *= 0.01
		if hotstart.size==(DIMS,1):
			print('New hotstart position defined.')
			sol = hotstart

		psi = matrix(0.0, (DIMS,N)) # (dim x exm)
		old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)
		threshold = 0

		obj = -1
		iter = 0 
		allobjs = []

		restarts = 0

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
				(foo, latent[i], psi[:,i]) = self.sobj.argmax(sol, i, add_prior=True)
				#print psi[:,i]
				#psi[:4,i] /= 600.0
				#psi[:,i] /= 600.0
				#psi[:4,i] = psi[:4,i]/np.linalg.norm(psi[:4,i],ord=2) 
				#psi[4:,i] = psi[4:,i]/np.linalg.norm(psi[4:,i],ord=2) 
				psi[:,i] /= np.linalg.norm(psi[:,i], ord=self.norm_ord)
				#psi[:,i] /= np.max(np.abs(psi[:,i]))
				#psi[:,i] /= 600.0
				#if i>10:
				#	(foo, latent[i], psi[:,i]) = self.sobj.argmax(sol,i)
				#else:
				#	psi[:,i] = self.sobj.get_joint_feature_map(i)
				#	latent[i] = self.sobj.y[i]
			print psi

			# 2. solve the intermediate convex optimization problem 
			kernel = Kernel.get_kernel(psi, psi)
			svm = OCSVM(kernel, self.C)
			svm.train_dual()
			threshold = svm.get_threshold()
			#inds = svm.get_support_dual()
			#alphas = svm.get_support_dual_values()
			#sol = phi[:,inds]*alphas

			self.svs_inds = svm.get_support_dual()
			#alphas = svm.get_support_dual_values()
			sol = psi*svm.get_alphas()
			print matrix([sol.trans(), old_sol.trans()]).trans()
			if len(self.svs_inds)==N and self.C>(1.0/float(N)):
				print('###################################')
				print('Degenerate solution.')
				print('###################################')

				restarts += 1
				if (restarts>10):
					print('###################################')
					print 'Too many restarts...'
					print('###################################')
					# calculate objective
					self.threshold = threshold
					slacks = [max([0.0, np.single(threshold - sol.trans()*psi[:,i]) ]) for i in xrange(N)]
					obj = 0.5*np.single(sol.trans()*sol) - np.single(threshold) + self.C*sum(slacks)
					print("Iter {0}: Values (Threshold-Slacks-Objective) = {1}-{2}-{3}".format(int(iter),np.single(threshold),np.single(sum(slacks)),np.single(obj)))
					allobjs.append(float(np.single(obj)))
					break

				# intermediate solutions
				# latent variables
				latent = [0.0]*N

				#setseed(0)
				sol = self.sobj.get_hotstart_sol()
				#sol[0:4] *= 0.01
				if hotstart.size==(DIMS,1):
					print('New hotstart position defined.')
					sol = hotstart

				psi = matrix(0.0, (DIMS,N)) # (dim x exm)
				old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)
				threshold = 0

				obj = -1
				iter = 0 
				allobjs = []

			# calculate objective
			self.threshold = threshold
			slacks = [max([0.0, np.single(threshold - sol.trans()*psi[:,i]) ]) for i in xrange(N)]
			obj = 0.5*np.single(sol.trans()*sol) - np.single(threshold) + self.C*sum(slacks)
			print("Iter {0}: Values (Threshold-Slacks-Objective) = {1}-{2}-{3}".format(int(iter),np.single(threshold),np.single(sum(slacks)),np.single(obj)))
			allobjs.append(float(np.single(obj)))

			# zero shot learning: single iteration, hence random
			# structure coefficient
			if zero_shot:
				print('LatentOcSvm: Zero shot learning.')
				break


		print '+++++++++'
		print threshold
		print slacks
		print obj
		print '+++++++++'
		self.slacks = slacks

		print allobjs
		print(sum(sum(abs(np.array(psi-old_psi)))))
		print '+++++++++ SAD END'		
		self.sol = sol
		self.latent = latent
		return (sol, latent, threshold)


	def apply(self, pred_sobj):
		""" Application of the StructuredOCSVM:

			anomaly_score = max_z <sol*,\Psi(x,z)> 
			latent_state = argmax_z <sol*,\Psi(x,z)> 
		"""
		N = pred_sobj.get_num_samples()
		vals = matrix(0.0, (N,1))
		structs = []
		for i in range(N):
			(vals[i], struct, psi) = pred_sobj.argmax(self.sol, i, add_prior=True)
			#psi[:] /= 600.0 
			#vals[i] = self.sol.trans()*psi - self.threshold
			#print np.multiply(self.sol,psi)
			#vals[i] *= -1.0
			vals[i] = (vals[i]/np.linalg.norm(psi, ord=self.norm_ord) - self.threshold)
			#vals[i] = self.sol[:4].trans()*psi[:4]/np.linalg.norm(psi[:4],ord=2) \
			#	+ self.sol[4:].trans()*psi[4:]/np.linalg.norm(psi[4:],ord=2) - self.threshold
			#vals[i] /= np.max(np.abs(psi))
			structs.append(struct)

		return (vals, structs)
