import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as io

from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA
from toydata import ToyData

from so_pgm import SOPGM

if __name__ == '__main__':
	# load data file
	data = io.loadmat('../ecoli/data.mat')
	exm_id_intervals = data['exm_id_intervals']
	exm_id = data['exm_id']
	label = data['label']
	signal = data['signal']
	
	EXMS = max(exm_id_intervals[:,0])
	EXMS = 100
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	# training data
	mean = 0.0
	cnt = 0 
	trainX = []
	trainY = []
	for i in xrange(EXMS):
		
		# convert signal to binary feature array
		# convert labels to states
		(foo,inds) = np.where([exm_id[0,:]==i])
		lens = len(inds)
		lbl = co.matrix(0, (1, lens))
		exm = co.sparse(co.matrix(0.0, (DIMS, lens)))
		for t in range(lens):
			exm[ int(np.int32(signal[0,inds[t]])), t ] = 1.0
			# labels to states
			val = max(0, label[0,inds[t]])
			lbl[t] = int(val)
		mean += co.matrix(1.0, (1, lens))*exm.trans()
		cnt += lens
		trainX.append(exm)
		trainY.append(lbl)

	mean = mean / float(cnt)
	print mean
	for i in range(EXMS):
		for d in range(DIMS):
			trainX[i][d,:] = trainX[i][d,:]-mean[d]

	# train
	pgm = SOPGM(trainX, trainY)
	lsvm = StructuredOCSVM(pgm, C=1.0/(EXMS*0.9))
	(lsol, lats, thres) = lsvm.train_dc(max_iter=20)

	# visualization
	plt.figure()
	for i in range(20):
		LENS = len(lats[i])
		plt.plot(range(LENS),lats[i].trans() + i*4,'-r')
		plt.plot(range(LENS),trainY[i].trans() + i*4,'-b')
		(anom_score, scores) = pgm.get_scores(lsol,i)
		plt.plot(range(LENS),scores.trans() + i*4,'-g')
	plt.show()

	print('finished')