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
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	DIST_LEN = 80
	distr1 = 10.0*np.logspace(-5,0,DIST_LEN)
	distr2 = 10.0*np.logspace(0,-5,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	# training data
	exm_cnt = 0
	mean = 0.0
	cnt = 0 
	trainX = []
	trainY = []
	for i in xrange(EXMS):
		
		# convert signal to binary feature array
		# convert labels to states
		#(foo,inds) = np.where([exm_id[0,:]==i])
		inds = range(exm_id_intervals[i,1]-1,exm_id_intervals[i,2])
		lens = len(inds)

		if lens>1000 or lens<=10:
			continue

		print('Index {0}: #{1}'.format(i,lens))
		exm_cnt += 1
		mod = 1
		lbl = co.matrix(0, (1, lens))
		exm = co.matrix(0.0, (DIMS, lens))
		for t in range(lens):
			start_ind = max(0,t-DIST_LEN)
			end_ind = min(lens-1,t+DIST_LEN-1)
			start = start_ind - (t-DIST_LEN)
			end = start + (end_ind-start_ind)

			#exm[ int(np.int32(signal[0,inds[t]])), t ] = 20.0
			exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] += distr[start:end]
			
			# labels to states
			val = max(0, label[0,inds[t]])
			if val==0 or val==1: 
				mod=1
			if val==3:
				lbl[t] = int(val + mod)
				mod = (mod+1) % 3
			else:
				lbl[t] = int(val)
		if exm_cnt==1:
			print exm 

		mean += co.matrix(1.0, (1, lens))*exm.trans()
		cnt += lens
		trainX.append(exm)
		trainY.append(lbl)

	# for i in range(60):
		
	# 	# convert signal to binary feature array
	# 	# convert labels to states
	# 	#(foo,inds) = np.where([exm_id[0,:]==i])
	# 	inds = range(2000+3*300,2000+3*300+300)
	# 	lens = len(inds)

	# 	if lens>300 or lens<=10:
	# 		continue

	# 	print('Index {0}: #{1}'.format(i,lens))
	# 	exm_cnt += 1
	# 	mod = 1
	# 	lbl = co.matrix(0, (1, lens))
	# 	exm = co.matrix(-1.0, (DIMS, lens))
	# 	for t in range(lens):
	# 		start_ind = max(0,t-DIST_LEN)
	# 		end_ind = min(lens-1,t+DIST_LEN-1)
	# 		start = start_ind - (t-DIST_LEN)
	# 		end = start + (end_ind-start_ind)

	# 		#exm[ int(np.int32(signal[0,inds[t]])), t ] = 20.0
	# 		exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] += distr[start:end]

	# 		# labels to states
	# 		val = max(0, label[0,inds[t]])
	# 		if val==0 or val==1: 
	# 			mod=1
	# 		if val==3:
	# 			lbl[t] = int(val + mod)
	# 			mod = (mod+1) % 3
	# 		else:
	# 			lbl[t] = int(val)

	# 	mean += co.matrix(1.0, (1, lens))*exm.trans()
	# 	cnt += lens
	# 	trainX.append(exm)
	# 	trainY.append(lbl)

	print exm_cnt
	EXMS = exm_cnt

	mean = mean / float(cnt)
	print mean
	#mean = co.matrix(0.0,(1,DIMS))
	for i in range(EXMS):
		for d in range(DIMS):
			trainX[i][d,:] = trainX[i][d,:]-mean[d]

	# train
	pgm = SOPGM(trainX, trainY)
	lsvm = StructuredOCSVM(pgm, C=1.0/(EXMS*0.5))
	lpca = StructuredPCA(pgm)
	ssvm = SSVM(pgm,C=1.0)
	(lsol, lats, thres) = lsvm.train_dc(max_iter=30)
	#(lsol, lats, thres) = lpca.train_dc(max_iter=20)
	#(lsol,slacks) = ssvm.train()
	#(vals, lats) = ssvm.apply(pgm)


	# visualization
	plt.figure()
	for i in range(20):
		LENS = len(lats[i])
		plt.plot(range(LENS),lats[i].trans() + i*8,'-r')
		plt.plot(range(LENS),trainY[i].trans() + i*8,'-b')
		
		(anom_score, scores) = pgm.get_scores(lsol, i, lats[i])
		plt.plot(range(LENS),scores.trans() + i*8,'-g')
	plt.show()

	print('finished')