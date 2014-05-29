import cvxopt as co
import numpy as np
import pylab as pl
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import scipy.io as io

from kernel import Kernel
from ocsvm import OCSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM

from toydata import ToyData
from so_hmm import SOHMM


def get_model(num_exm, num_train, lens, block_len, blocks=1, anomaly_prob=0.15):
	print('Generating {0} sequences, {1} for training, each with {2} anomaly probability.'.format(num_exm, num_train, anomaly_prob))
	cnt = 0 
	X = [] 
	Y = []
	label = []
	for i in range(num_exm):
		(exm, lbl, marker) = ToyData.get_2state_anom_seq(lens, block_len, anom_prob=anomaly_prob, num_blocks=blocks)
		cnt += lens
		X.append(exm)
		Y.append(lbl)
		label.append(marker)
	X = remove_mean(X,1)
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)


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

def calc_feature_vecs(data):
	# ASSUME that all sequences have the same length!
	N = len(data)
	(F, LEN) = data[0].size
	phi = co.matrix(0.0, (F*LEN, N))
	for i in xrange(N):
		for f in xrange(F):
			phi[(f*LEN):(f*LEN)+LEN,i] = data[i][f,:].trans()

		norm = np.linalg.norm(phi[:,i],2)
		#print norm
		phi[:,i] /= norm

	return phi  


def experiment_anomaly_detection(train, test, comb, num_train, anom_prob, labels):
	phi = calc_feature_vecs(comb.X)
	print phi.size

	# bayes classifier
	(DIMS, N) = phi.size
	w_bayes = co.matrix(1.0, (DIMS, 1))
	pred = w_bayes.trans()*phi[:,num_train:]
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred.trans())
	bayes_auc = metric.auc(fpr, tpr)

	# train one-class svm
	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train])
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc = metric.auc(fpr, tpr)
	if (base_auc<0.5):
	    base_auc = 1.0-base_auc

	# train structured anomaly detection
	sad = StructuredOCSVM(train, C=1.0/(num_train*anom_prob))
	#sad = StructuredOCSVM(train, C=1.0/(num_train*0.5))
	(lsol, lats, thres) = sad.train_dc(max_iter=50)
	(pred_vals, pred_lats) = sad.apply(test)
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred_vals)
	auc = metric.auc(fpr, tpr)
	if (auc<0.5):
	    auc = 1.0-auc

	return (auc, base_auc, bayes_auc)


if __name__ == '__main__':
	LENS = 600
	EXMS = 1000
	EXMS_TRAIN = 200
	ANOM_PROB = 0.05
	REPS = 10
	BLOCK_LEN = 100
	#BLOCKS = [1]
	BLOCKS = [1,5,10,25,100,200]

	# collected means
	mauc = []
	mbase_auc = [] 
	mbayes_auc = [] 

	# collected variances
	vauc = []
	vbase_auc = [] 
	vbayes_auc = [] 
	for b in xrange(len(BLOCKS)):
		aucs = []
		fmauc = 0.0
		fmbase_auc = 0.0 
		fmbayes_auc = 0.0 
		for r in xrange(REPS):
			(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, BLOCK_LEN, blocks=BLOCKS[b], anomaly_prob=ANOM_PROB)
			(auc, base_auc, bayes_auc) = experiment_anomaly_detection(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
			aucs.append((auc, base_auc, bayes_auc))
			fmauc += auc
			fmbase_auc += base_auc
			fmbayes_auc += bayes_auc

		mauc.append(fmauc/float(REPS))
		mbase_auc.append(fmbase_auc/float(REPS))
		mbayes_auc.append(fmbayes_auc/float(REPS))
		vauc.append(sum([ (aucs[i][0]-mauc[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc.append(sum([ (aucs[i][1]-mbase_auc[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbayes_auc.append(sum([ (aucs[i][2]-mbayes_auc[b])**2 for i in xrange(REPS)]) / float(REPS))


	print '####################'
	print('Mean/Variance    SAD={0} / {1}'.format(mauc, vauc))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc, vbase_auc))
	print('Mean/Variance  BAYES={0} / {1}'.format(mbayes_auc, vbayes_auc))
	print '####################'

	# store result as a file
	data = {}
	data['LENS'] = LENS
	data['EXMS'] = EXMS
	data['EXMS_TRAIN'] = EXMS_TRAIN
	data['ANOM_PROB'] = ANOM_PROB
	data['REPS'] = REPS
	data['BLOCKS'] = BLOCKS

	data['mauc'] = mauc
	data['mbase_auc'] = mbase_auc
	data['mbayes_auc'] = mbayes_auc
	data['vauc'] = vauc
	data['vbase_auc'] = vbase_auc
	data['vbayes_auc'] = vbayes_auc

	io.savemat('14_nips_toy_ad_08.mat',data)

	print('finished')
