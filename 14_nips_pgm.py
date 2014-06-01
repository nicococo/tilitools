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



def get_example(signal, label, dims, start_pos, end_pos):
	""" This method converts the signal, label vector into 
		a feature matrix, a state vector and a spectrum vector
		for a single example.
	"""
	(foo, LEN) = label.shape

	DIST_LEN = 160
	distr1 = 1.0*np.logspace(-6,0,DIST_LEN)
	distr2 = 1.0*np.logspace(0,-6,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	# adjust end position to not lie in a genic region
	while ((label[0, end_pos]>-1 or label[0, end_pos-20]>-1) and end_pos<=LEN):
		end_pos += 1 

	inds = range(start_pos, end_pos)
	lens = len(inds)

	mod = 1 # modulo counter (0,1,2) for inner exon states
	lbl = co.matrix(0, (1, lens))
	exm = co.matrix(0.0, (dims, lens))
	phi = co.matrix(0.0, (1, dims))

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
		if val==3:
			lbl[t] = int(val + mod)
			mod = (mod+1) % 3
			containsGene = True
		else:
			lbl[t] = int(val)

		# spectrum kernel entry
		phi[int(np.int32(signal[0,inds[t]]))] +=1.0/float(lens)
	return (exm, lbl, phi, containsGene, end_pos)


def get_example_list(num, dims, signal, label, start, min_lens=600, max_lens=800):
	(foo, LEN) = label.shape
	min_genes = int(float(num)*0.15)

	X = []
	Y = []
	phi = []
	marker = []

	cnt_genes = 0
	cnt = 0
	while (cnt<num):
		lens = np.int(np.single(co.uniform(1, a=min_lens, b=max_lens)))
		if (start+lens)>LEN:
			print('Warning! End of genome. Could not add example.')
			break
		(exm, lbl, phi_i, isGene, end_pos) = get_example(signal, label, dims, start, start+lens)
		
		# accept example, if it has the correct length
		if (end_pos-start<=max_lens or (isGene==True and end_pos-start<800)):		
			X.append(exm)
			Y.append(lbl)
			phi.append(phi_i)
			if isGene:
				marker.append(0)
				cnt_genes += 1
				min_genes -= 1
			else:
				marker.append(1)
			cnt += 1
		start = end_pos

	print('Number of examples {0}. {1} of them are genic.'.format(len(Y), cnt_genes))
	return (X, Y, phi, marker, start) 


def get_model(num_exm, num_train):
	num_test = num_exm-num_train

	# load data file
	data = io.loadmat('../ecoli/data.mat')
	exm_id_intervals = data['exm_id_intervals']
	exm_id = data['exm_id']
	label = data['label']
	signal = data['signal']
	
	EXMS = max(exm_id_intervals[:,0])
	DIMS = 4**3
	print('There are {0} gene examples.'.format(EXMS))

	#start = exm_id_intervals[100,1]
	start = 0
	MIN_LEN = 600
	MAX_LEN = 610

	# 1. load training examples
	(trainX, trainY, phi1, marker1, stop) = get_example_list(num_train, DIMS, signal, label, start, min_lens=MIN_LEN, max_lens=MAX_LEN)


	cnt = 0
	gene_cnt = 0
	phi = []
	while (gene_cnt<80):
		ind = cnt

		start = exm_id_intervals[ind,1]
		stop = exm_id_intervals[ind,2]
		if (stop-start>=800):
			cnt += 1
			continue

		(exm, lbl, phi_i, isGene, end_pos) = get_example(signal, label, DIMS, start, stop)

		# accept example, if it has the correct length
		if (end_pos-start<=800):		
			trainX.append(exm)
			trainY.append(lbl)
			phi.append(phi_i)
			if isGene:
				marker1.append(0)
			else:
				marker1.append(1)
			gene_cnt += 1
		cnt += 1
	print cnt
	print gene_cnt
	trainX = remove_mean(trainX, DIMS)

	# 2. load test examples
	(testX, testY, phi2, marker2, stop) = get_example_list(num_test, DIMS, signal, label, stop, min_lens=MIN_LEN, max_lens=MAX_LEN)

	combX = list(trainX)
	combX.append(list(testX))
	combY = list(trainY)
	combY.append(list(testY))
	return (SOPGM(trainX,trainY), SOPGM(testX,testY), SOPGM(combX,combY), marker1, co.matrix(phi1).trans())
	#return (SOHMM(trainX,trainY), SOHMM(testX,testY), SOHMM(combX,combY), marker1, co.matrix(phi1).trans())



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


if __name__ == '__main__':
	EXMS_TRAIN = 80
	EXMS_TEST = 100
	EXMS_COMB = EXMS_TRAIN+EXMS_TEST

	(train, test, comb, label, phi) = get_model(EXMS_COMB, EXMS_TRAIN)
	lsvm = StructuredOCSVM(train, C=1.0/(train.samples*0.5))
	(lsol, lats, thres) = lsvm.train_dc(max_iter=100)
	(err, err_exm) = train.evaluate(lats)
	print err

	plt.figure()
	scores = []
	for i in range(train.samples):
	 	LENS = len(train.y[i])
	 	(anom_score, scores_exm) = train.get_scores(lsol, i, lats[i])
	 	scores.append(anom_score)
	 	plt.plot(range(LENS),scores_exm.trans() + i*8,'-g')

	 	plt.plot(range(LENS),train.y[i].trans() + i*8,'-b')
	 	plt.plot(range(LENS),lats[i].trans() + i*8,'-r')

	 	if i==0:
		 	plt.plot(range(LENS),train.X[i][0,:].trans() + i*8,'-k')


	(fpr, tpr, thres) = metric.roc_curve(label, co.matrix(scores))
	auc = metric.auc(fpr, tpr)
	print auc

	# train one-class svm
	kern = Kernel.get_kernel(phi, phi)
	ocsvm = OCSVM(kern, C=1.0/(EXMS_TRAIN*0.1))
	ocsvm.train_dual()
	(oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(label[0:EXMS_TRAIN], oc_as)
	base_auc = metric.auc(fpr, tpr)
	print base_auc

	plt.show()

	print('finished')