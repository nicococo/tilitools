import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as io
import sklearn.metrics as metric

from kernel import Kernel
from ocsvm import OCSVM

from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA
from toydata import ToyData

from so_pgm import SOPGM


def add_intergenic(num_exm, signal, label, region_start, region_end, exm_lens, distr):
	trainX = []
	trainY = []
	phi = []
	DIST_LEN = np.int(np.ceil(float(len(distr))/2.0))
	for i in range(num_exm):
		inds = range(region_start+i*exm_lens,region_start+(i+1)*exm_lens)
		lens = len(inds)

		print('Index {0}: #{1}'.format(i,lens))
		phi_i = co.matrix(0.0, (1, DIMS))
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
			# spectrum kernel entry
			phi_i[int(np.int32(signal[0,inds[t]]))] +=1.0/float(lens)

			# labels to states
			val = max(0, label[0,inds[t]])
			error += val
			lbl[t] = int(val)

		phi.append(phi_i)

		if (error>0):
			print 'ERROR loading integenic regions: gene found!'
		trainX.append(exm)
		trainY.append(lbl)
	return (trainX, trainY, phi)


def find_intergenic_regions(labels, min_gene_dist=20):
	(foo, N) = labels.shape
	total_len = 0
	cnt = 0
	ige_intervals = []
	start = 0
	stop = 0
	isCnt = True
	for t in xrange(N):
		if (isCnt==True and labels[0,t]>0):
			isCnt = False
			stop = t-1
			dist = stop-start
			if dist>2*min_gene_dist:
				ige_intervals.append((start+min_gene_dist, stop-min_gene_dist))
				cnt += 1
				total_len += dist-2*min_gene_dist
		if (isCnt==False and labels[0,t]<0):
			start = t
			isCnt = True
	print('Found {0} intergenic regions with a total length of {1}.'.format(cnt, total_len))
	return ige_intervals


def remove_mean(X, dims):
	cnt = 0
	tst_mean = co.matrix(0.0, (1, dims))
	for i in range(len(X)):
		lens = len(X[i][0,:])
		cnt += lens
		tst_mean += co.matrix(1.0, (1, lens))*X[i].trans()
	tst_mean /= float(cnt)
	print tst_mean
	for i in range(len(X)):
		for d in range(dims):
			X[i][d,:] = X[i][d,:]-tst_mean[d]
	cnt = 0
	tst_mean = co.matrix(0.0, (1, dims))
	for i in range(len(X)):
		lens = len(X[i][0,:])
		cnt += lens
		tst_mean += co.matrix(1.0, (1, lens))*X[i].trans()
	print tst_mean/float(cnt)
	return X


def load_genes(max_genes, signal, label, exm_id_intervals, distr, min_lens=600, max_lens=800):
	DIMS = 4**3
	EXMS = len(exm_id_intervals[:,0])
	DIST_LEN = np.int(np.ceil(float(len(distr))/2.0))

	# training data
	trainX = []
	trainY = []
	start_symbs = []
	stop_symbs = []
	phi_list = []
	marker = []
	for i in xrange(EXMS):
		# convert signal to binary feature array
		# convert labels to states
		#(foo,inds) = np.where([exm_id[0,:]==i])
		inds = range(exm_id_intervals[i,1]-1,exm_id_intervals[i,2])
		lens = len(inds)

		if lens>max_lens or lens<=min_lens:
			continue

		max_genes -= 1
		if max_genes<0:
			break

		print('Index {0}: #{1}'.format(i,lens))
		mod = 1
		lbl = co.matrix(0, (1, lens))
		exm = co.matrix(0.0, (DIMS, lens))
		phi_i = co.matrix(0.0, (1, DIMS))
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

			# spectrum kernel entry
			phi_i[int(np.int32(signal[0,inds[t]]))] +=1.0/float(lens)

		marker.append(0)
		phi_list.append(phi_i)
		trainX.append(exm)
		trainY.append(lbl)

	print '###################'
	print start_symbs
	print stop_symbs 
	print '###################'
	return (trainX, trainY, phi_list, marker)


def load_intergenics(num_iges, signal, label, ige_intervals, distr, min_lens=600, max_lens=800):
	# add intergenic examples
	marker = []
	trainX = []
	trainY = []
	phi_list = []
	ige_cnt = 0
	IGE_EXMS = num_iges
	N = len(ige_intervals)
	for i in xrange(N):
		lens = ige_intervals[i][1]-ige_intervals[i][0]
		if lens>10000:
			IGE_LEN = np.int(np.single(co.uniform(1, a=min_lens, b=max_lens)))
			num_ige_exms = np.int(np.floor(float(lens)/float(IGE_LEN)))
			if (num_ige_exms > IGE_EXMS-ige_cnt):
				num_ige_exms = IGE_EXMS-ige_cnt
			ige_cnt += num_ige_exms
			
			(X, Y, phis) = add_intergenic(num_ige_exms, signal, label, ige_intervals[i][0], ige_intervals[i][1], IGE_LEN, distr)
			trainX.extend(X)
			trainY.extend(Y)
			phi_list.extend(phis)

			for j in range(num_ige_exms):
				marker.append(1)
		if ige_cnt>IGE_EXMS:
			break
	print('IGE examples {0}'.format(ige_cnt))
	return (trainX, trainY, phi_list, marker)


if __name__ == '__main__':
	# load data file
	data = io.loadmat('../ecoli/data.mat')
	exm_id_intervals = data['exm_id_intervals']
	exm_id = data['exm_id']
	label = data['label']
	signal = data['signal']

	# find intergenic regions
	ige_intervals = find_intergenic_regions(label)
	IGE_REGIONS = len(ige_intervals)
	
	EXMS = max(exm_id_intervals[:,0])
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	DIST_LEN = 160
	distr1 = 1.0*np.logspace(-5,0,DIST_LEN)
	distr2 = 1.0*np.logspace(0,-5,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	NUM_TRAIN_GEN = 20
	NUM_TRAIN_IGE = 80
	
	NUM_TEST_GEN = 10
	NUM_TEST_IGE = 40

	NUM_COMB_GEN = NUM_TRAIN_GEN+NUM_TEST_GEN
	NUM_COMB_IGE = NUM_TRAIN_IGE+NUM_TEST_IGE

	REPS = 5

	auc = []
	base_auc = []
	res = []
	base_res = []
	for r in xrange(REPS):
		# shuffle genes and intergenics
		inds = np.random.permutation(EXMS)
		exm_id_intervals = exm_id_intervals[inds,:]
		ige_intervals = np.random.permutation(ige_intervals)

		# load genes and intergenic examples
		(combX, combY, phi_list, marker) = load_genes(NUM_COMB_GEN, signal, label, exm_id_intervals, distr, min_lens=600, max_lens=1000)
		(X, Y, phis, lbls) = load_intergenics(NUM_COMB_IGE, signal, label, ige_intervals, distr, min_lens=600, max_lens=800)
		combX.extend(X)
		combY.extend(Y)
		phi_list.extend(phis)
		marker.extend(lbls)
		EXMS = len(combY)
		combX = remove_mean(combX, DIMS)

		trainX = combX[0:NUM_TRAIN_GEN]
		trainX.extend(X[0:NUM_TRAIN_IGE])
		trainY = combY[0:NUM_TRAIN_GEN]
		trainY.extend(Y[0:NUM_TRAIN_IGE])

		testX = combX[NUM_TRAIN_GEN:]
		testX.extend(X[NUM_TRAIN_IGE:])
		testY = combY[NUM_TRAIN_GEN:]
		testY.extend(Y[NUM_TRAIN_IGE:])

		train = SOPGM(trainX, trainY)
		test = SOPGM(testX, testY)
		comb = SOPGM(combX, combY)

		# SSVM annotation
		ssvm = SSVM(train, C=10.0)
		(lsol,slacks) = ssvm.train()
		(vals, svmlats) = ssvm.apply(test)
		(err_svm, err_exm) = test.evaluate(svmlats)
		base_res.append((err_svm['fscore'], err_svm['precision'], err_svm['sensitivity'], err_svm['specificity']))
		print err_svm

		# SAD annotation
		lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*0.5))
		(lsol, lats, thres) = lsvm.train_dc(max_iter=100)
		(lval, lats) = lsvm.apply(test)
		(err, err_exm) = test.evaluate(lats)
		res.append((err['fscore'], err['precision'], err['sensitivity'], err['specificity']))
		print err

		# SAD anomaly scores
		(scores, foo) = lsvm.apply(comb)
		(fpr, tpr, thres) = metric.roc_curve(marker, scores)
		auc.append(metric.auc(fpr, tpr))
		print auc

		# train one-class svm
		phi = co.matrix(phi_list).trans()
		kern = Kernel.get_kernel(phi, phi)
		ocsvm = OCSVM(kern, C=1.0/(comb.samples*0.25))
		ocsvm.train_dual()
		(oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
		(fpr, tpr, thres) = metric.roc_curve(marker, oc_as)
		base_auc.append(metric.auc(fpr, tpr))
		print base_auc

	
	print '##############################################'
	print auc
	print base_auc
	print '##############################################'
	print res
	print base_res
	print '##############################################'

	# store result as a file
	data = {}
	data['auc'] = auc
	data['base_auc'] = base_auc
	data['res'] = res
	data['base_res'] = base_res

	io.savemat('14_nips_pgm_01.mat',data)

	# ssvm = SSVM(pgm, C=1.0)
	# (lsol,slacks) = ssvm.train()
	# (vals, svmlats) = ssvm.apply(pgm)
	# (err_svm, err_exm) = pgm.evaluate(svmlats)
	# print err
	# print err_rnd
	# print err_zeros
	# print err_svm
	# print err_lowest

	# # training data
	# exm_cnt = 0
	# trainX = []
	# trainY = []
	# start_symbs = []
	# stop_symbs = []
	# phi_list = []
	# marker = []
	# for i in xrange(EXMS):
		
	# 	# convert signal to binary feature array
	# 	# convert labels to states
	# 	#(foo,inds) = np.where([exm_id[0,:]==i])
	# 	inds = range(exm_id_intervals[i,1]-1,exm_id_intervals[i,2])
	# 	lens = len(inds)

	# 	if lens>800 or lens<=600:
	# 		continue

	# 	print('Index {0}: #{1}'.format(i,lens))
	# 	exm_cnt += 1
	# 	mod = 1
	# 	lbl = co.matrix(0, (1, lens))
	# 	exm = co.matrix(0.0, (DIMS, lens))
	# 	phi_i = co.matrix(0.0, (1, DIMS))
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

	# 		# store start/stop symbols
	# 		if (label[0,inds[t]]==1):
	# 			start_symbs.append(signal[0,inds[t]])
	# 		if (label[0,inds[t]]==2):
	# 			stop_symbs.append(signal[0,inds[t]])

	# 		# spectrum kernel entry
	# 		phi_i[int(np.int32(signal[0,inds[t]]))] +=1.0/float(lens)

	# 	marker.append(0)
	# 	phi_list.append(phi_i)
	# 	trainX.append(exm)
	# 	trainY.append(lbl)


	# print '###################'
	# print start_symbs
	# print stop_symbs 
	# print '###################'

	# # add intergenic examples
	# ige_cnt = 0
	# IGE_EXMS = 105
	# IGE_LEN = 600
	# N = len(ige_intervals)
	# for i in xrange(N):
	# 	lens = ige_intervals[i][1]-ige_intervals[i][0]
	# 	if lens>1000:
	# 		IGE_LEN = np.int(np.single(co.uniform(1, a=600, b=800)))
	# 		num_ige_exms = np.int(np.floor(float(lens)/float(IGE_LEN)))
	# 		if (num_ige_exms > IGE_EXMS):
	# 			num_ige_exms = IGE_EXMS - ige_cnt + 1
	# 		exm_cnt += num_ige_exms
	# 		ige_cnt += num_ige_exms
	# 		(trainX, trainY, phi_list) = add_intergenic(trainX, trainY, phi_list, ige_intervals[i][0], ige_intervals[i][1], num_ige_exms,IGE_LEN,distr,DIST_LEN)
	# 		for j in range(num_ige_exms):
	# 			marker.append(1)
	# 	if ige_cnt>IGE_EXMS:
	# 		break

	# print('IGE examples {0}'.format(ige_cnt))

	# phi = co.matrix(phi_list).trans()
	# print phi.size
	# print exm_cnt
	# EXMS = exm_cnt

	# trainX = remove_mean(trainX, DIMS)

	# # train
	# pgm = SOPGM(trainX, trainY)
	# (err_lowest, fscore_exm) = pgm.evaluate(trainY)
	# print err_lowest

	# base_zeros = []
	# base_rnd = []
	# for i in xrange(EXMS):
	# 	lens = len(trainY[i])
	# 	foo = co.matrix(0, (1,lens))
	# 	base_zeros.append(foo)
	# 	base_rnd.append(np.round(co.uniform(1, lens)))

	# (err_rnd, fscore_exm) = pgm.evaluate(base_rnd)
	# (err_zeros, fscore_exm) = pgm.evaluate(base_zeros)
	# print err_rnd
	# print err_zeros

	# lsvm = StructuredOCSVM(pgm, C=1.0/(EXMS*0.5))
	# lpca = StructuredPCA(pgm)
	# (lsol, lats, thres) = lsvm.train_dc(max_iter=100)
	
	# ssvm = SSVM(pgm, C=1.0)
	# (lsol,slacks) = ssvm.train()
	# (vals, svmlats) = ssvm.apply(pgm)
	# (err_svm, err_exm) = pgm.evaluate(svmlats)
	# (err, err_exm) = pgm.evaluate(lats)
	# print err
	# print err_rnd
	# print err_zeros
	# print err_svm
	# print err_lowest

	# # visualization
	# plt.figure()
	# allscores = []
	# for i in range(pgm.samples):
	# 	LENS = len(lats[i])
	# 	plt.plot(range(LENS),lats[i].trans() + i*8,'-r')
	# 	plt.plot(range(LENS),trainY[i].trans() + i*8,'-b')
		
	# 	(anom_score, scores) = pgm.get_scores(lsol, i, lats[i])
	# 	allscores.append(anom_score)
	# 	plt.plot(range(LENS),scores.trans() + i*8,'-g')

	# (fpr, tpr, thres) = metric.roc_curve(marker, co.matrix(allscores))
	# auc = metric.auc(fpr, tpr)
	# print auc

	# # train one-class svm
	# kern = Kernel.get_kernel(phi, phi)
	# ocsvm = OCSVM(kern, C=1.0/(pgm.samples*0.4))
	# ocsvm.train_dual()
	# (oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
	# (fpr, tpr, thres) = metric.roc_curve(marker, oc_as)
	# base_auc = metric.auc(fpr, tpr)
	# print base_auc


	# plt.show()



	print('finished')