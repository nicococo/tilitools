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
		#phi[:,i] /= norm

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

	# oc-svm without pre-processing
	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train])
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc = metric.auc(fpr, tpr)

	# normalize data
	for idx in range(phi.size[1]):
		phi[:,idx] /= np.linalg.norm(phi[:,idx])

	# train one-class svm
	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train])
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc1 = metric.auc(fpr, tpr)

	# train one-class svm RBF
	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=0.1)
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi, type='rbf', param=0.1)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc2 = metric.auc(fpr, tpr)

	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=1.0)
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi, type='rbf', param=1.0)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc3 = metric.auc(fpr, tpr)

	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=2.0)
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi, type='rbf', param=2.0)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc4 = metric.auc(fpr, tpr)

	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=4.0)
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi, type='rbf', param=4.0)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc5 = metric.auc(fpr, tpr)

	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=10.0)
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi, type='rbf', param=10.0)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc6 = metric.auc(fpr, tpr)

	# train structured anomaly detection
	sad = StructuredOCSVM(train, C=1.0/(num_train*anom_prob))
	(lsol, lats, thres) = sad.train_dc(max_iter=60)
	(pred_vals, pred_lats) = sad.apply(test)
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred_vals)
	auc = metric.auc(fpr, tpr)
	return (auc, base_auc, bayes_auc, base_auc1, base_auc2, base_auc3, base_auc4, base_auc5, base_auc6)


if __name__ == '__main__':
	LENS = 600
	EXMS = 1000
	EXMS_TRAIN = 200
	ANOM_PROB = 0.15
	REPS = 20
	BLOCK_LEN = 100
	#BLOCKS = [1,100]
	BLOCKS = [1,2,5,10,20,40,60,80,100]
	#BLOCKS = [1,2,10]

	# collected means
	mauc = []
	mbase_auc = [] 
	mbayes_auc = [] 

	mbase_auc1 = [] 
	mbase_auc2 = [] 
	mbase_auc3 = [] 
	mbase_auc4 = [] 
	mbase_auc5 = [] 
	mbase_auc6 = [] 

	# collected variances
	vauc = []
	vbase_auc = [] 
	vbayes_auc = [] 

	vbase_auc1 = [] 
	vbase_auc2 = [] 
	vbase_auc3 = [] 
	vbase_auc4 = [] 
	vbase_auc5 = [] 
	vbase_auc6 = [] 

	for b in xrange(len(BLOCKS)):
		aucs = []
		fmauc = 0.0
		fmbase_auc = 0.0 
		fmbayes_auc = 0.0 

		fmbase_auc1 = 0.0 
		fmbase_auc2 = 0.0 
		fmbase_auc3 = 0.0 
		fmbase_auc4 = 0.0 
		fmbase_auc5 = 0.0 
		fmbase_auc6 = 0.0 
		for r in xrange(REPS):
			(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, BLOCK_LEN, blocks=BLOCKS[b], anomaly_prob=ANOM_PROB)
			(auc, base_auc, bayes_auc, base_auc1, base_auc2, base_auc3, base_auc4, base_auc5, base_auc6)  = experiment_anomaly_detection(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
			aucs.append((auc, base_auc, bayes_auc, base_auc1, base_auc2, base_auc3, base_auc4, base_auc5, base_auc6))
			fmauc += auc
			fmbase_auc += base_auc
			fmbayes_auc += bayes_auc

			fmbase_auc1 += base_auc1
			fmbase_auc2 += base_auc2
			fmbase_auc3 += base_auc3
			fmbase_auc4 += base_auc4
			fmbase_auc5 += base_auc5
			fmbase_auc6 += base_auc6


		mauc.append(fmauc/float(REPS))
		mbase_auc.append(fmbase_auc/float(REPS))
		mbayes_auc.append(fmbayes_auc/float(REPS))

		mbase_auc1.append(fmbase_auc1/float(REPS))
		mbase_auc2.append(fmbase_auc2/float(REPS))
		mbase_auc3.append(fmbase_auc3/float(REPS))
		mbase_auc4.append(fmbase_auc4/float(REPS))
		mbase_auc5.append(fmbase_auc5/float(REPS))
		mbase_auc6.append(fmbase_auc6/float(REPS))

		vauc.append(sum([ (aucs[i][0]-mauc[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc.append(sum([ (aucs[i][1]-mbase_auc[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbayes_auc.append(sum([ (aucs[i][2]-mbayes_auc[b])**2 for i in xrange(REPS)]) / float(REPS))

		vbase_auc1.append(sum([ (aucs[i][3]-mbase_auc1[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc2.append(sum([ (aucs[i][4]-mbase_auc2[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc3.append(sum([ (aucs[i][5]-mbase_auc3[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc4.append(sum([ (aucs[i][6]-mbase_auc4[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc5.append(sum([ (aucs[i][7]-mbase_auc5[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc6.append(sum([ (aucs[i][8]-mbase_auc6[b])**2 for i in xrange(REPS)]) / float(REPS))


	print '####################'
	print('Mean/Variance    SAD={0} / {1}'.format(mauc, vauc))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc, vbase_auc))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc1, vbase_auc1))
	print('Mean/Variance  BAYES={0} / {1}'.format(mbayes_auc, vbayes_auc))
	print '####################'
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc2, vbase_auc2))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc3, vbase_auc3))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc4, vbase_auc4))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc5, vbase_auc5))
	print('Mean/Variance  OCSVM={0} / {1}'.format(mbase_auc6, vbase_auc6))
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

	data['mbase_auc1'] = mbase_auc1
	data['mbase_auc2'] = mbase_auc2
	data['mbase_auc3'] = mbase_auc3
	data['mbase_auc4'] = mbase_auc4
	data['mbase_auc5'] = mbase_auc5
	data['mbase_auc6'] = mbase_auc6

	data['vbase_auc1'] = vbase_auc1
	data['vbase_auc2'] = vbase_auc2
	data['vbase_auc3'] = vbase_auc3
	data['vbase_auc4'] = vbase_auc4
	data['vbase_auc5'] = vbase_auc5
	data['vbase_auc6'] = vbase_auc6

	io.savemat('15_icml_toy_ad_03.mat',data)

	print('finished')
