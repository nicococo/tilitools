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



def get_example(signal, label, dims, start_pos, end_pos):
	""" This method converts the signal, label vector into 
		a feature matrix, a state vector and a spectrum vector
		for a single example.
	"""
	(foo, LEN) = label.shape

	DIST_LEN = 160
	distr1 = 1.0*np.logspace(-5,0,DIST_LEN)
	distr2 = 1.0*np.logspace(0,-5,DIST_LEN)
	distr = np.concatenate([distr1, distr2[1:]])

	# adjust end position to not lie in a genic region
	while ((label[0, end_pos]>-1 or label[0, end_pos-20]>-1) and end_pos<=LEN):
		end_pos += 1 

	inds = range(start_pos-1, end_pos)
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

	mean = co.matrix(1.0, (1, lens))*exm.trans() / float(lens)
	return (exm, mean, lbl, phi, containsGene, end_pos)


def get_example_list(num, dims, signal, label, start, min_lens=600, max_lens=800):
	(foo, LEN) = label.shape
	min_genes = int(float(num)*0.15)

	X = []
	Y = []
	phi = []
	mean = co.matrix(0.0, (dims, 1))
	marker = []

	cnt_genes = 0
	cnt = 0
	while (cnt<num):
		lens = np.int(np.single(co.uniform(1, a=min_lens, b=max_lens)))
		if (start+lens)>LEN:
			print('Warning! End of genome. Could not add example.')
			break
		(exm, mean_i, lbl, phi_i, isGene, end_pos) = get_example(signal, label, dims, start, start+lens)
		
		# accept example, if it has the correct length
		if (end_pos-start<=max_lens or (isGene==True and end_pos-start<1000)):		
			mean += mean_i.trans()
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
	return (X, Y, mean/float(len(Y)), phi, marker, start) 


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

	start = exm_id_intervals[100,1]
	start = 0
	MIN_LEN = 800
	MAX_LEN = 810

	# 1. load training examples
	(trainX, trainY, mean1, phi1, marker1, stop) = get_example_list(num_train, DIMS, signal, label, start, min_lens=MIN_LEN, max_lens=MAX_LEN)

	# 2. load test examples
	(testX, testY, mean2, phi2, marker2, stop) = get_example_list(num_test, DIMS, signal, label, stop, min_lens=MIN_LEN, max_lens=MAX_LEN)

	combX = list(trainX)
	combX.append(list(testX))
	combY = list(trainY)
	combY.append(list(testY))
	return (SOPGM(trainX,trainY), SOPGM(testX,testY), SOPGM(combX,combY), marker1, co.matrix(phi1).trans())



def remove_mean(X, dims, mean):
	for i in range(len(X)):
		for d in range(DIMS):
			trainX[i][d,:] = trainX[i][d,:]-mean[d]
	return X


if __name__ == '__main__':
	EXMS_TRAIN = 70
	EXMS_TEST = 100
	EXMS_COMB = EXMS_TRAIN+EXMS_TEST

	(train, test, comb, label, phi) = get_model(EXMS_COMB, EXMS_TRAIN)
	lsvm = StructuredOCSVM(train, C=1.0/(EXMS_TRAIN*0.9))
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

	(fpr, tpr, thres) = metric.roc_curve(label, co.matrix(scores))
	auc = metric.auc(fpr, tpr)
	print auc

	# train one-class svm
	kern = Kernel.get_kernel(phi, phi)
	ocsvm = OCSVM(kern, C=1.0/(EXMS_TRAIN*0.05))
	ocsvm.train_dual()
	(oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(label, oc_as)
	base_auc = metric.auc(fpr, tpr)
	print base_auc

	plt.show()

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