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
from so_hmm import SOHMM

def smooth(x,window_len=4,window='blackman'):
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."

	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."


	if window_len<3:
		return x

	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

	s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y


def add_intergenic(num_exm, signal, label, region_start, region_end, exm_lens, distr):
	trainX = []
	trainY = []
	phi = []
	DIMS = 4**3
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
			foo = distr[start:end] + exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ]
			exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] = foo[0]
			# spectrum kernel entry
			phi_i[int(np.int32(signal[0,inds[t]]))] +=1.0/float(lens)

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
			foo = distr[start:end] + exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ]
			exm[ int(np.int32(signal[0,inds[t]])), start_ind:end_ind ] = foo[0]

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
	#data = io.loadmat('/home/nico/Data/data.mat')
	data = io.loadmat('/home/nicococo/Code/ecoli/data.mat')
	#data = io.loadmat('/home/nicococo/Code/anthracis/data.mat')
	exm_id_intervals = data['exm_id_intervals']
	exm_id = data['exm_id']
	label = data['label']
	signal = data['signal']

	# find intergenic regions
	#ige_intervals = find_intergenic_regions(label, min_gene_dist=2)
	#intervals = {}
	#intervals['ige'] = ige_intervals
	#io.savemat('../ecoli/ige_intervals_2.mat',intervals)
	
	intervals = io.loadmat('../ecoli/ige_intervals_50.mat')
	ige_intervals = intervals['ige']

	IGE_REGIONS = len(ige_intervals)
	
	EXMS = max(exm_id_intervals[:,0])
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	DIST_LEN = 100
	distr1 = 1.0*np.logspace(-4,0,DIST_LEN)
	distr2 = 1.0*np.logspace(0,-4,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	NUM_TRAIN_GEN = 10
	NUM_TRAIN_IGE = 60
	
	NUM_TEST_GEN = 10
	NUM_TEST_IGE = 60

	NUM_COMB_GEN = NUM_TRAIN_GEN+NUM_TEST_GEN
	NUM_COMB_IGE = NUM_TRAIN_IGE+NUM_TEST_IGE

	REPS = 1

	showPlots = False

	auc = []
	base_auc = []
	res = []
	base_res = []
	for r in xrange(REPS):
		# shuffle genes and intergenics
		inds = np.random.permutation(EXMS)
		#exm_id_intervals = exm_id_intervals[inds,:]
		exm_id_intervals = np.random.permutation(exm_id_intervals)
		ige_intervals = np.random.permutation(ige_intervals)

		# load genes and intergenic examples
		(combX, combY, phi_list, marker) = load_genes(NUM_COMB_GEN, signal, label, exm_id_intervals, distr, min_lens=600, max_lens=800)
		#exm_id_intervals = np.random.permutation(exm_id_intervals)		
		#(X, Y, phis, lbls) = load_genes(NUM_COMB_IGE, signal, label, exm_id_intervals, distr, min_lens=600, max_lens=800)
		(X, Y, phis, lbls) = load_intergenics(NUM_COMB_IGE, signal, label, ige_intervals, distr, min_lens=600, max_lens=800)
		combX.extend(X)
		combY.extend(Y)
		phi_list.extend(phis)
		marker.extend(lbls)
		EXMS = len(combY)
		combX = remove_mean(combX, DIMS)

		# get rid of unnessary dims
		phi = co.matrix(phi_list).trans()
		mgenes = np.sum(phi[:,1:NUM_COMB_GEN],1)/NUM_COMB_GEN
		mige = np.sum(phi[:,NUM_COMB_GEN:],1)/NUM_COMB_IGE

		mige /= max(abs(mige))
		mgenes /= max(abs(mgenes))
		print mige
		print len(mige)

		inds = np.argsort(mgenes)
		igeinds = np.argsort(mige)
		print inds
		#plt.plot(range(64),mgenes[inds] - 0,'-r')
		#plt.plot(range(64),mige[inds] - 2,'-b')
		#plt.show()

		# keep only the most informativ gene dims
		inds1 = inds[56:]
		inds2 = inds[0:8]

		inds3 = igeinds[0:8]
		inds4 = igeinds[56:]

		inds5 = np.array([0,1,2,63,62,61,60])
		#inds6 = 
		#inds5 = np.array([0,1])
		#inds6 = np.array([63,62,61,60])
		print inds
		for i in range(len(combX)):
			foo1 = co.matrix(np.sum(combX[i][inds1.tolist(),:],0)/float(len(inds1.tolist())))
			foo2 = co.matrix(np.sum(combX[i][inds2.tolist(),:],0)/float(len(inds2.tolist())))
			foo3 = co.matrix(np.sum(combX[i][inds3.tolist(),:],0)/float(len(inds3.tolist())))
			foo4 = co.matrix(np.sum(combX[i][inds4.tolist(),:],0)/float(len(inds4.tolist())))
			
			foo5 = co.matrix(np.sum(combX[i][inds5.tolist(),:],0)/float(len(inds5.tolist())))
			
			foo6 = foo1-foo4
			
			#combX[i] = co.matrix([foo1.trans(), foo2.trans(), foo3.trans(), foo4.trans(), foo5.trans(), foo6.trans()])
			#print combX[i].size
		#DIMS = 6

		#combX = remove_mean(combX, len(inds.tolist()))
		combX = remove_mean(combX, DIMS)


		trainX = combX[0:NUM_TRAIN_GEN]
		trainX.extend(X[0:NUM_TRAIN_IGE])
		trainY = combY[0:NUM_TRAIN_GEN]
		trainY.extend(Y[0:NUM_TRAIN_IGE])

		testX = combX[NUM_TRAIN_GEN:NUM_COMB_GEN]
		testX.extend(X[NUM_TRAIN_IGE:NUM_COMB_IGE])
		testY = combY[NUM_TRAIN_GEN:NUM_COMB_GEN]
		testY.extend(Y[NUM_TRAIN_IGE:NUM_COMB_IGE])

		state_map = []
		#state_map.append([3])
		#state_map.append([4])
		#state_map.append([4])
		#state_map.append([0])
		#state_map.append([0])
		#state_map.append([0])

		state_map.append(range(64))
		state_map.append([0,1,2,3])
		state_map.append([63,62,61,60,59])
		state_map.append(range(64))
		state_map.append(range(64))
		state_map.append(range(64))

		train = SOPGM(trainX, trainY, state_dims_map=state_map)
		test = SOPGM(testX, testY, state_dims_map=state_map)
		comb = SOPGM(combX, combY, state_dims_map=state_map)

		#train = SOPGM(trainX, trainY, state_dims_map=[])
		#test = SOPGM(testX, testY, state_dims_map=[])
		#comb = SOPGM(combX, combY, state_dims_map=[])


		#for i in range(len(combY)):
		#	combY[i] = co.matrix(np.sign(combY[i]), tc='i')
		#for i in range(len(trainY)):
		#	trainY[i] = co.matrix(np.sign(trainY[i]), tc='i')
		#for i in range(len(testY)):
		#	testY[i] = co.matrix(np.sign(testY[i]), tc='i')

		#print len(trainY)
		#print trainY[0]
		#train = SOHMM(trainX, trainY)
		#test = SOHMM(testX, testY)
		#comb = SOHMM(combX, combY)

		# SSVM annotation
		#ssvm = SSVM(train, C=10.0)
		#(lsol,slacks) = ssvm.train()
		#(vals, svmlats) = ssvm.apply(test)
		#(err_svm, err_exm) = test.evaluate(svmlats)
		#base_res.append((err_svm['fscore'], err_svm['precision'], err_svm['sensitivity'], err_svm['specificity']))
		base_res.append((0.0,0.0,0.0,0.0))

		# SAD annotation
		lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*0.15))
		(lsol, lats, thres) = lsvm.train_dc(max_iter=100)

		if (showPlots==True):
			#for i in range(comb.samples):
			for i in range(5):
				#if (marker[i]==0):
				LENS = len(comb.y[i])
				for d in range(DIMS):
					plt.plot(range(LENS),comb.X[i][d,:].trans() - 2*d+(i-10)*10,'-m')

				plt.plot(range(LENS),lats[i].trans() +(i-10)*10,'-r')
				plt.plot(range(LENS),comb.y[i].trans() + 2 +(i-10)*10,'-b')
		
				(anom_score, scores) = comb.get_scores(lsol, i, lats[i])
				plt.plot(range(LENS),scores.trans() + 6 + (i-10)*10,'-g')
				#plt.show()

		(lval, lats) = lsvm.apply(test)
		(err, err_exm) = test.evaluate(lats)
		res.append((err['fscore'], err['precision'], err['sensitivity'], err['specificity']))
		print err
		#print err_svm

		# SAD anomaly scores
		(scores, foo) = lsvm.apply(comb)
		(fpr, tpr, thres) = metric.roc_curve(marker, scores)
		cur_auc = metric.auc(fpr, tpr)
		if (cur_auc<0.5):
			cur_auc = 1.0-cur_auc
		auc.append(cur_auc)
		print auc

		# train one-class svm
		phi = co.matrix(phi_list).trans()
		kern = Kernel.get_kernel(phi, phi)
		ocsvm = OCSVM(kern, C=1.0/(comb.samples*0.15))
		ocsvm.train_dual()
		(oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
		(fpr, tpr, thres) = metric.roc_curve(marker, oc_as)
		cur_auc = metric.auc(fpr, tpr)
		if (cur_auc<0.5):
			cur_auc = 1.0-cur_auc
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

	io.savemat('14_nips_pgm_03.mat',data)

	print('finished')