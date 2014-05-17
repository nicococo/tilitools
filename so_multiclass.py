from cvxopt import matrix,spmatrix,sparse
import numpy as np
import math as math
from so_interface import SOInterface


class SOMultiClass(SOInterface):
	""" Multi class structured object."""

	num_classes = -1 # (scalar) number of classes 


	def __init__(self, X, classes, y=[]):
		# the class also acts as indices therefore:
		# y >= 0!
		SOInterface.__init__(self, X, y)
		self.num_classes = classes		


	def argmax(self, sol, idx, add_loss=False, opt_type='linear'):
		nd = self.dims
		d = 0  # start of dimension in sol
		val = -10**10
		cls = -1 # best class

		for c in range(self.num_classes):
			foo = sol[d:d+nd].trans()*self.X[:,idx]
			# the argmax of the above function
			# is equal to the argmax of the quadratic function
			# foo = + 2*foo - normPsi
			# since ||\Psi(x_i,z)|| = ||\phi(x_i)|| = y \forall z   
			d += nd
			if (np.single(foo)>np.single(val)):
				val = foo
				cls = c

		if (opt_type=='quadratic'):
			normPsi = self.X[:,idx].trans()*self.X[:,idx]
			val = 2*val - normPsi

		psi_idx = self.get_joint_feature_map(idx,cls)
		return (val, cls, psi_idx)

		
	def logsumexp(self, sol, idx, add_loss=False, opt_type='linear'):
		nd = self.dims
		d = 0  # start of dimension in sol
		val = 0.0
		for c in xrange(self.num_classes):
			val += exp(sol[d:d+nd].trans()*self.X[:,idx])
			d += nd
		return log(val)


	def calc_loss(self, idx, y):
		return self.y[idx]!=y


	def get_joint_feature_map(self, idx, y=-1):
		if y==-1:
			y=self.y[idx]

		nd = self.dims
		mc = self.num_classes
		psi = matrix(0.0,(nd*mc,1))
		psi[nd*y:nd*(y+1)] = self.X[:,idx]
		return psi

	def get_num_dims(self):
		return self.dims*self.num_classes