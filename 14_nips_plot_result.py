import cvxopt as co
import numpy as np
import pylab as pl
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import scipy.io as io

from kernel import Kernel
from ocsvm import OCSVM
from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA

from toydata import ToyData
from so_hmm import SOHMM


if __name__ == '__main__':

	FNAME = '14_nips_toy_anno_02.mat'
	data = io.loadmat(FNAME)

	conts = data['conts']
	conts_base = data['conts_base']

	N = len(conts[0,:])
	M = len(conts[0,0][0,0])

	for m in range(M):
		res_sad = []
		res_ssvm = []
		for i in range(N):
			res_sad.append(conts[0,i][0,0][m][0][0])
			res_ssvm.append(conts_base[0,i][0,0][m][0][0])
 		plt.plot(range(N),res_sad,'-k')
 		plt.plot(range(N),res_ssvm,'-r')

	plt.show()

	print('finished')