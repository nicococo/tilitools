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

	@staticmethod
	def get_2state_gaussian_seq(lens,dims=2,means1=[2,2],means2=[5,5],vars1=[1,1],vars2=[1,1],anom_prob=1.0):
		
		seqs = co.matrix(0.0, (dims, lens))
		lbls = co.matrix(0, (1,lens))
		marker = 0

		# generate first state sequence
		for d in range(dims):
			seqs[d,:] = co.normal(1,lens)*vars1[d] + means1[d]

		prob = np.random.uniform()
		if prob<anom_prob:		
			# add second state blocks
			while (True):
				max_block_len = 0.6*lens
				min_block_len = 0.1*lens
				block_len = np.int(max_block_len*np.single(co.uniform(1))+3)
				block_start = np.int(lens*np.single(co.uniform(1)))

				if (block_len - (block_start+block_len-lens)-3>min_block_len):
					break

			block_len = min(block_len,block_len - (block_start+block_len-lens)-3)
			lbls[block_start:block_start+block_len-1] = 1
			marker = 1
			for d in range(dims):
				#print block_len
				seqs[d,block_start:block_start+block_len-1] = co.normal(1,block_len-1)*vars2[d] + means2[d]

		return (seqs, lbls, marker)