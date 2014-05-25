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


# def add_intergenic(trainX, trainY, mean, region_start, region_end, num_exm, exm_lens, distr, DIST_LEN):
# 	for i in range(num_exm):
# 		inds = range(region_start+i*exm_lens,region_start+(i+1)*exm_lens)
# 		lens = len(inds)

# 		print('Index {0}: #{1}'.format(i,lens))
# 		error = 0
# 		mod = 1
# 		lbl = co.matrix(0, (1, lens))
# 		exm = co.matrix(-1.0, (DIMS, lens))
# 		for t in range(lens):
# 			start_ind = max(0,t-DIST_LEN)
# 			end_ind = min(lens-1,t+DIST_LEN-1)
# 			start = start_ind - (t-DIST_LEN)
# 			end = start + (end_ind-start_ind)

# 			#exm[ int(np.int32(signal[0,inds[t]])), t ] = 20.0
# 			exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] += distr[start:end]
# 			# labels to states
# 			val = max(0, label[0,inds[t]])
# 			error += val
# 			lbl[t] = int(val)

# 		if (error>0):
# 			print 'ERROR loading integenic regions: gene found!'
# 		mean += co.matrix(1.0, (1, lens))*exm.trans()
# 		trainX.append(exm)
# 		trainY.append(lbl)
# 	return (trainX, trainY, mean)


def get_example(signal, label, start_pos, end_pos):
	""" This method converts the signal, label vector into 
		a feature matrix, a state vector and a spectrum vector
		for a single example.
	"""
	DIST_LEN = 160
	distr1 = 1.0*np.logspace(-5,0,DIST_LEN)
	distr2 = 1.0*np.logspace(0,-5,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	inds = range(start_pos-1, end_pos)
	lens = len(inds)

	mod = 1 # modulo counter (0,1,2) for inner exon states
	lbl = co.matrix(0, (1, lens))
	exm = co.matrix(0.0, (DIMS, lens))
	phi = co.matrix(0.0, (1, DIMS))

	containsGene = False
	for t in range(lens):
		# calc start and end positions fr the distr-vector
		start_ind = max(0,t-DIST_LEN)
		end_ind = min(lens-1,t+DIST_LEN-1)
		start = start_ind - (t-DIST_LEN)
		end = start + (end_ind-start_ind)

		exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] += distr[start:end]
		
		# labels to states
		val = max(0, label[0,inds[t]])
		if val==0 or val==1: 
			mod=1
			containsGene = True
		if val==3:
			lbl[t] = int(val + mod)
			mod = (mod+1) % 3
		else:
			lbl[t] = int(val)

		# spectrum kernel entry
		phi[int(np.int32(signal[0,inds[t]]))] +=1.0/float(lens)

	mean = co.matrix(1.0, (1, lens))*exm.trans() / float(lens)
	return (exm, mean, lbl, phi, containsGene)


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


def get_intergenic_examples(num, dims, signal, label, ige_intervals, min_lens=600, max_lens=800):
	# add intergenic examples
	X = []
	Y = []
	phi = []
	mean = co.matrix(0.0, (dims, 1))
	ige_cnt = 0
	N = len(ige_intervals)
	for i in xrange(N):
		lens = ige_intervals[i][1]-ige_intervals[i][0]
		if lens>10000:
			IGE_LEN = np.int(np.single(co.uniform(1, a=min_lens, b=max_lens)))
			num_ige_exms = np.int(np.floor(float(lens)/float(IGE_LEN)))
			if (num_ige_exms > num):
				num_ige_exms = num - ige_cnt

			ige_cnt += num_ige_exms	
			for r in xrange(num_ige_exms):
	 			start = r*IGE_LEN + ige_intervals[i][0]
	 			stop = start + IGE_LEN
				(exm, mean_i, lbl, phi_i, isGene) = get_example(signal, label, start, stop)
				mean += mean_i
				X.append(exm)
				Y.append(lbl)
				phi.append(phi_i)
				if isGene:
					print('Warning! Intergenic region contains gene!')

		if ige_cnt>=num:
			break
	print('IGE examples {0}'.format(ige_cnt))
	return (X, Y, mean/float(ige_cnt), phi.trans(), i) 


def get_model(num_exm, num_train, anomaly_prob=0.15):
	num_test = num_exm-num_train
	
	num_exm_train_gen = int(np.round(num_train * anomaly_prob))
	num_exm_train_ige = num_train-num_exm_train_gen 
	
	num_exm_test_gen = int(np.round(num_test * anomaly_prob))
	num_exm_test_ige = num_test-num_exm_test_gen 

	# load data file
	data = io.loadmat('../ecoli/data.mat')
	exm_id_intervals = data['exm_id_intervals']
	exm_id = data['exm_id']
	label = data['label']
	signal = data['signal']

	# find intergenic regions
	ige_intervals = find_intergenic_regions(label)
	
	EXMS = max(exm_id_intervals[:,0])
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	# 1. load training examples
	(train_ige_X, train_ige_Y, mean1, phi) = get_intergenic_examples(num_exm_train_ige, DIMS, signal, label, ige_intervals, min_lens=600, max_lens=800):

	# 2. load test examples


	combX = list(trainxX)
	combX.append(list(testX))
	combY = list(trainY)
	combY.append(list(testY))
	return (SOPGM(trainX,trainY), SOPGM(testX,testY), SOPGM(combX,combY), label, phi)



def experiment():

	return 0

if __name__ == '__main__':

	(train, test, comb, label, phi) = get_model(400, 100, anomaly_prob=0.15)


	# # load data file
	# data = io.loadmat('../ecoli/data.mat')
	# exm_id_intervals = data['exm_id_intervals']
	# exm_id = data['exm_id']
	# label = data['label']
	# signal = data['signal']

	# # find intergenic regions
	# ige_intervals = find_intergenic_regions(label)
	
	# EXMS = max(exm_id_intervals[:,0])
	# DIMS = 4**3
	# print('There are {0} gene examples.'.format(EXMS))

	# DIST_LEN = 160
	# distr1 = 1.0*np.logspace(-5,0,DIST_LEN)
	# distr2 = 1.0*np.logspace(0,-5,DIST_LEN)
	# distr = np.concatenate([distr1, distr2[1:]])

	# # training data
	# exm_cnt = 0
	# mean = 0.0
	# cnt = 0 
	# trainX = []
	# trainY = []
	# start_symbs = []
	# stop_symbs = []
	# phi_list = []
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

	# 	phi_list.append(phi_i)
	# 	if exm_cnt==1:
	# 		print exm 

	# 	mean += co.matrix(1.0, (1, lens))*exm.trans()
	# 	cnt += lens
	# 	trainX.append(exm)
	# 	trainY.append(lbl)


	# phi = co.matrix(phi_list)
	# print phi.size

	# print '###################'
	# print start_symbs
	# print stop_symbs 
	# print '###################'

	# # add intergenic examples
	# ige_cnt = 0
	# IGE_EXMS = 70
	# IGE_LEN = 600
	# N = len(ige_intervals)
	# for i in xrange(N):
	# 	lens = ige_intervals[i][1]-ige_intervals[i][0]
	# 	if lens>10000:
	# 		IGE_LEN = np.int(np.single(co.uniform(1, a=600, b=800)))
	# 		num_ige_exms = np.int(np.floor(float(lens)/float(IGE_LEN)))
	# 		if (num_ige_exms > IGE_EXMS):
	# 			num_ige_exms = IGE_EXMS - ige_cnt+1

	# 		exm_cnt += num_ige_exms
	# 		cnt += num_ige_exms*IGE_LEN
	# 		ige_cnt += num_ige_exms
	# 		(trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, ige_intervals[i][0], ige_intervals[i][1], num_ige_exms,IGE_LEN,distr,DIST_LEN)
	# 	if ige_cnt>IGE_EXMS:
	# 		break

	# print('IGE examples {0}'.format(ige_cnt))

	# # exm_lens = 600
	# # num = 12
	# # exm_cnt += num
	# # cnt += num*exm_lens
	# # (trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, 8500, 16700, num, exm_lens,distr,DIST_LEN)

	# # exm_lens = 600
	# # num = 5
	# # exm_cnt += num
	# # cnt += num*exm_lens
	# # (trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, 4700, 7600, num, exm_lens,distr,DIST_LEN)

	# # exm_lens = 600
	# # num = 26 #57
	# # exm_cnt += num
	# # cnt += num*exm_lens
	# # (trainX, trainY, mean) = add_intergenic(trainX, trainY, mean, 44400, 62000, num, exm_lens,distr,DIST_LEN)

	# print exm_cnt
	# EXMS = exm_cnt

	# mean = mean / float(cnt)
	# print mean
	# #mean = co.matrix(0.0,(1,DIMS))
	# for i in range(EXMS):
	# 	for d in range(DIMS):
	# 		trainX[i][d,:] = trainX[i][d,:]-mean[d]

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

	# # load hotstart file
	# #old_sol = []
	# #with open('../pgm_hotstart1.csv', 'rb') as f:
	# #	old_sol = np.loadtxt(f, delimiter=',')
	# #print old_sol

	# lsvm = StructuredOCSVM(pgm, C=1.0/(EXMS*0.5))
	# lpca = StructuredPCA(pgm)
	# (lsol, lats, thres) = lsvm.train_dc(max_iter=100)
	# #(lsol, lats, thres) = lpca.train_dc(max_iter=20)
	
	# #ssvm = SSVM(pgm, C=1.0)
	# #(lsol,slacks) = ssvm.train()
	# #(vals, svmlats) = ssvm.apply(pgm)
	# #(err_svm, err_exm) = pgm.evaluate(svmlats)
	# (err, err_exm) = pgm.evaluate(lats)
	# print err
	# print err_rnd
	# print err_zeros
	# #print err_svm
	# print err_lowest

	# # visualization
	# plt.figure()
	# for i in range(70):
	# 	LENS = len(lats[i])
	# 	plt.plot(range(LENS),lats[i].trans() + i*8,'-r')
	# 	plt.plot(range(LENS),trainY[i].trans() + i*8,'-b')
		
	# 	(anom_score, scores) = pgm.get_scores(lsol, i, lats[i])
	# 	plt.plot(range(LENS),scores.trans() + i*8,'-g')
	# plt.show()

	print('finished')