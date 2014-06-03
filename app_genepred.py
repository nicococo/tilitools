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


def feature_selection(fname, top_k=32, num_exms=100):
	""" Simple feature selection technique:
		(a) choose a different but related organism (e.g. e.fergusonii for e.coli)
		(b) load a certain number of genes and intergenic regions
		(c) use the top_k (=highest mean) of the spectrum feature vectors for IGE and gen
	"""
	# load genes and intergenic examples
	(foo1, foo2, phi_list, foo3) = load_genes(num_exms, signal, label, exm_id_intervals, distr, min_lens=600, max_lens=800)
	phi = co.matrix(phi_list).trans()
	mgenes = np.sum(phi,1)/float(num_exms)
	mgenes /= max(abs(mgenes))

	(foo1, foo2, phi_list, foo3) = load_intergenics(num_exms, signal, label, ige_intervals, distr, min_lens=600, max_lens=800)
	phi = co.matrix(phi_list).trans()
	mige = np.sum(phi,1)/float(num_exms)
	mige /= max(abs(mige))

	# sort indices: most frequent or the last ones
	gen_inds = np.argsort(mgenes)
	ige_inds = np.argsort(mige)
	print('Calculated filter indices from {0} with {1} examples.'.format(fname,num_exms))
	print gen_inds
	print ige_inds
	
	inds = gen_inds[64-top_k:]
	inds = np.append(inds, ige_inds[64-top_k:])
	phi_inds = np.unique(inds)
	print phi_inds.shape
	print phi_inds

	#inds = {}
	#inds['ige'] = ige_inds
	#inds['gen'] = gen_inds
	#io.savemat('fergusonii_discr_feats.mat',inds)
	return (gen_inds[64-top_k:], ige_inds[64-top_k:], phi_inds)


if __name__ == '__main__':
	#data = io.loadmat('/home/nico/Data/data.mat')
	data = io.loadmat('/home/nicococo/Code/ecoli/data.mat')
	#data = io.loadmat('/home/nicococo/Code/anthracis/data.mat')
	#data = io.loadmat('/home/nicococo/Code/fergusonii/data.mat')
	exm_id_intervals = data['exm_id_intervals']
	exm_id = data['exm_id']
	label = data['label']
	signal = data['signal']

	# output file
	out_fname = '14_nips_pgm_03.mat'

	# find intergenic regions
	#ige_intervals = find_intergenic_regions(label, min_gene_dist=50)
	#intervals = {}
	#intervals['ige'] = ige_intervals
	#io.savemat('../ecoli/ige_intervals_2.mat',intervals)
	
	intervals = io.loadmat('../ecoli/ige_intervals_50.mat')
	ige_intervals = intervals['ige']

	IGE_REGIONS = len(ige_intervals)
	
	EXMS = max(exm_id_intervals[:,0])
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	DIST_LEN = 160
	distr1 = 1.0*np.logspace(-6,0,DIST_LEN)
	distr2 = 1.0*np.logspace(0,-6,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	NUM_TRAIN_GEN = 20
	NUM_TRAIN_IGE = 100
	
	NUM_TEST_GEN = 20
	NUM_TEST_IGE = 100

	NUM_COMB_GEN = NUM_TRAIN_GEN+NUM_TEST_GEN
	NUM_COMB_IGE = NUM_TRAIN_IGE+NUM_TEST_IGE
	anom_prob = float(NUM_COMB_GEN)/float(NUM_COMB_GEN+NUM_COMB_IGE)

	REPS = 1

	showPlots = True

	(gen_inds, ige_inds, phi_inds) = feature_selection('/home/nicococo/Code/fergusonii/data.mat', top_k=24)

	auc = []
	base_auc = []
	res = []
	base_res = []
	for r in xrange(REPS):
		# shuffle genes and intergenics
		exm_id_intervals = np.random.permutation(exm_id_intervals)
		ige_intervals = np.random.permutation(ige_intervals)

		# load genes and intergenic examples
		(combX, combY, phi_list, marker) = load_genes(NUM_COMB_GEN, signal, label, exm_id_intervals, distr, min_lens=600, max_lens=800)
		(X, Y, phis, lbls) = load_intergenics(NUM_COMB_IGE, signal, label, ige_intervals, distr, min_lens=600, max_lens=800)
		combX.extend(X)
		combY.extend(Y)
		phi_list.extend(phis)
		marker.extend(lbls)
		EXMS = len(combY)
		combX = remove_mean(combX, DIMS)

		total_len = 0
		for i in range(EXMS):
			total_len += len(combY[i])
		print('---> Total length = {0}.'.format(total_len))

		trainX = combX[0:NUM_TRAIN_GEN]
		trainX.extend(X[0:NUM_TRAIN_IGE])
		trainY = combY[0:NUM_TRAIN_GEN]
		trainY.extend(Y[0:NUM_TRAIN_IGE])

		testX = combX[NUM_TRAIN_GEN:NUM_COMB_GEN]
		testX.extend(X[NUM_TRAIN_IGE:NUM_COMB_IGE])
		testY = combY[NUM_TRAIN_GEN:NUM_COMB_GEN]
		testY.extend(Y[NUM_TRAIN_IGE:NUM_COMB_IGE])

		state_map = []
		state_map.append(ige_inds.tolist())
		state_map.append([0,1,2,3])
		state_map.append([63,62,61,60,59])
		state_map.append(gen_inds.tolist())
		state_map.append(gen_inds.tolist())
		state_map.append(gen_inds.tolist())

		train = SOPGM(trainX, trainY, state_dims_map=state_map)
		test = SOPGM(testX, testY, state_dims_map=state_map)
		comb = SOPGM(combX, combY, state_dims_map=state_map)

		# SSVM annotation
		#ssvm = SSVM(train, C=10.0)
		#(lsol,slacks) = ssvm.train()
		#(vals, svmlats) = ssvm.apply(test)
		#(err_svm, err_exm) = test.evaluate(svmlats)
		#base_res.append((err_svm['fscore'], err_svm['precision'], err_svm['sensitivity'], err_svm['specificity']))
		base_res.append((0.0,0.0,0.0,0.0))

		# SAD annotation
		lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*anom_prob))
		(lsol, lats, thres) = lsvm.train_dc(max_iter=100)

		if (showPlots==True):
			#for i in range(comb.samples):
			for i in range(36,45):
				#if (marker[i]==0):
				LENS = len(comb.y[i])
				for d in phi_inds.tolist():
					plt.plot(range(LENS),comb.X[i][d,:].trans() - 2*d+(i-10)*10,'-m')

				plt.plot(range(LENS),lats[i].trans() +(i-10)*10,'-r')
				plt.plot(range(LENS),comb.y[i].trans() + 2 +(i-10)*10,'-b')
		
				(anom_score, scores) = comb.get_scores(lsol, i, lats[i])
				plt.plot(range(LENS),scores.trans() + 6 + (i-10)*10,'-g')
				plt.show()

		(lval, lats) = lsvm.apply(test)
		(err, err_exm) = test.evaluate(lats)
		res.append((err['fscore'], err['precision'], err['sensitivity'], err['specificity']))
		print err
		print base_res[r]

		# SAD anomaly scores
		(scores, foo) = lsvm.apply(comb)
		(fpr, tpr, thres) = metric.roc_curve(marker, scores)
		cur_auc = metric.auc(fpr, tpr)
		if (cur_auc<0.5):
			cur_auc = 1.0-cur_auc
		auc.append(cur_auc)
		print auc

		# train one-class svm (use only filtered features)
		phi = co.matrix(phi_list).trans()
		phi = phi[phi_inds.tolist(),:]
		kern = Kernel.get_kernel(phi, phi)
		ocsvm = OCSVM(kern, C=1.0/(comb.samples*anom_prob))
		ocsvm.train_dual()
		(oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
		(fpr, tpr, thres) = metric.roc_curve(marker, oc_as)
		cur_auc = metric.auc(fpr, tpr)
		if (cur_auc<0.5):
			cur_auc = 1.0-cur_auc
		base_auc.append(cur_auc)
		print base_auc

	print '##############################################'
	print  out_fname
	print '##############################################'
	print NUM_COMB_GEN
	print NUM_COMB_IGE
	print '##############################################'
	print total_len	
	print anom_prob
	print '##############################################'
	print ige_inds
	print gen_inds	
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

	io.savemat(out_fname, data)
	print('finished')