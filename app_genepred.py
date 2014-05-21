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


def add_intergenic(trainX, trainY, mean, region_start, region_end, num_exm, exm_lens, distr, DIST_LEN):
	for i in range(num_exm):
		inds = range(region_start+i*exm_lens,region_start+(i+1)*exm_lens)
		lens = len(inds)

		print('Index {0}: #{1}'.format(i,lens))
		error = 0
		mod = 1
		lbl = co.matrix(0, (1, lens))
		exm = co.matrix(-1.0, (DIMS, lens))
		for t in range(lens):
			start_ind = max(0,t-DIST_LEN)
			end_ind = min(lens-1,t+DIST_LEN-1)
			start = start_ind - (t-DIST_LEN)
			end = start + (end_ind-start_ind)

			#exm[ int(np.int32(signal[0,inds[t]])), t ] = 20.0
			exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] += distr[start:end]
			# labels to states
			val = max(0, label[0,inds[t]])
			error += val
			lbl[t] = int(val)

		if (error>0):
			print 'ERROR loading integenic regions: gene found!'
		mean += co.matrix(1.0, (1, lens))*exm.trans()
		trainX.append(exm)
		trainY.append(lbl)
	return (trainX, trainY, mean)


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

	DIST_LEN = 160
	distr1 = 20.0*np.logspace(-5,0,DIST_LEN)
	distr2 = 20.0*np.logspace(0,-5,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	# training data
	exm_cnt = 0
	mean = 0.0
	cnt = 0 
	trainX = []
	trainY = []
	start_symbs = []
	stop_symbs = []
	for i in xrange(EXMS):
		
		# convert signal to binary feature array
		# convert labels to states
		#(foo,inds) = np.where([exm_id[0,:]==i])
		inds = range(exm_id_intervals[i,1]-1,exm_id_intervals[i,2])
		lens = len(inds)

		if lens>300 or lens<=10:
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

			# store start/stop symbols
			if (label[0,inds[t]]==1):
				start_symbs.append(signal[0,inds[t]])
			if (label[0,inds[t]]==2):
				stop_symbs.append(signal[0,inds[t]])

		if exm_cnt==1:
			print exm 

		mean += co.matrix(1.0, (1, lens))*exm.trans()
		cnt += lens
		trainX.append(exm)
		trainY.append(lbl)

	print start_symbs
	print stop_symbs 

	exm_lens = 300
	num = 25
	exm_cnt += num
	cnt += num*exm_lens
	(trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, 8500, 16700, num, exm_lens,distr,DIST_LEN)

	exm_lens = 300
	num = 10
	exm_cnt += num
	cnt += num*exm_lens
	(trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, 4700, 7600, num, exm_lens,distr,DIST_LEN)

	exm_lens = 300
	num = 7 #57
	exm_cnt += num
	cnt += num*exm_lens
	(trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, 44400, 62000, num, exm_lens,distr,DIST_LEN)

	# for i in range(100):
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
	(fscore, fscore_exm) = pgm.evaluate(trainY)
	print fscore/float(EXMS)

	base_zeros = []
	base_rnd = []
	for i in xrange(EXMS):
		lens = len(trainY[i])
		foo = co.matrix(0, (1,lens))
		base_zeros.append(foo)
		base_rnd.append(np.round(co.uniform(1, lens)))

	(fscore, fscore_exm) = pgm.evaluate(base_rnd)
	print fscore/float(EXMS)

	(fscore, fscore_exm) = pgm.evaluate(base_zeros)
	print fscore/float(EXMS)


	lsvm = StructuredOCSVM(pgm, C=1.0/(EXMS*0.5))
	lpca = StructuredPCA(pgm)
	ssvm = SSVM(pgm,C=1.0)
	(lsol, lats, thres) = lsvm.train_dc(max_iter=200)
	#(lsol, lats, thres) = lpca.train_dc(max_iter=20)
	#(lsol,slacks) = ssvm.train()
	#(vals, lats) = ssvm.apply(pgm)

	(fscore, fscore_exm) = pgm.evaluate(lats)
	print fscore/float(EXMS)

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