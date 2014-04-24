import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from ssvm import SSVM
from so_multiclass import SOMultiClass

class ToyData:


	@staticmethod
	def get_gaussian(num,dims=2,means=[0,0],vars=[1,1]):
		data = co.matrix(0.0,(dims,num))
		for d in range(dims):
			data[d,:] = co.normal(1,num)*vars[d] + means[d]
		return data