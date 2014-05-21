import cvxopt as co
import numpy as np
import pylab as pl
import sklearn.metrics as metric
import matplotlib.pyplot as plt

from kernel import Kernel
from ocsvm import OCSVM
from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA

from toydata import ToyData
from so_hmm import SOHMM


def get_model(num_exm, num_train, lens, feats, anomaly_prob=0.15):
	print('Generating {0} sequences, {1} for training, each with {2} anomaly probability.'.format(num_exm, num_train, anomaly_prob))
	mean = 0.0
	cnt = 0 
	X = [] 
	Y = []
	label = []
	for i in range(num_exm):
		(exm, lbl, marker) = ToyData.get_2state_gaussian_seq(lens,dims=feats,anom_prob=anomaly_prob)
		#if i<4:
		#	(exm,lbl) = ToyData.get_2state_gaussian_seq(LENS,dims=2,means1=[1,-3],means2=[3,7],vars1=[1,1],vars2=[1,1])
		mean += co.matrix(1.0, (1, lens))*exm.trans()
		cnt += lens
		X.append(exm)
		Y.append(lbl)
		label.append(marker)

	mean = mean / float(cnt)
	for i in range(num_exm):
		#if (i<10):
		#	pos = int(np.single(co.uniform(1))*float(LENS)*0.8 + 4.0)
		#	print pos
		#	trainX[i][1,pos] = 100.0
		for d in range(feats):
			X[i][d,:] = X[i][d,:]-mean[d]
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)


def calc_feature_vecs(data):
	# ASSUME that all sequences have the same length!
	N = len(data)
	(F, LEN) = data[0].size

	phi = co.matrix(0.0, (F*LEN, N))
	for i in xrange(N):
		for f in xrange(F):
			phi[(f*LEN):(f*LEN)+LEN,i] = data[i][f,:].trans()
	return phi  


if __name__ == '__main__':
	DIMS = 2
	LENS = 250
	EXMS = 200
	EXMS_TRAIN = 100

	# generate data
	(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, DIMS)
	(phi) = calc_feature_vecs(comb.X)

	ssvm = SSVM(train)
	(sol, slacks) = ssvm.train()

	kern = Kernel.get_kernel(phi, phi)
	ocsvm = OCSVM(kern, C=1.0/(EXMS*0.15))
	ocsvm.train_dual()
	(oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
	(fpr,tpr,thres) = metric.roc_curve(labels, oc_as)
	auc = metric.auc(fpr, tpr)
 	print '#########################'
	print auc 


	lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*0.5))
	(lsol, lats, thres) = lsvm.train_dc(max_iter=40)

	# test
	(vals, pred) = ssvm.apply(test)

	plt.figure()
	for d in range(DIMS):
		plt.plot(range(LENS),test.X[0][d,:].trans() + 5.0*d,'-r')
	
	(anom_score, scores) = test.get_scores(lsol, 0, pred[0])
	plt.plot(range(LENS),scores.trans() + 5.0,'-b')

	plt.plot(range(LENS),pred[0].trans() - 5,'-g')
	plt.plot(range(LENS),lats[EXMS_TRAIN].trans() - 7.5,'-r')
	plt.plot(range(LENS),test.y[0].trans() - 10,'-b')
	plt.show()

	for i in range(20):
		plt.plot(range(LENS),lats[i].trans() + i*4,'-r')
		plt.plot(range(LENS),comb.y[i].trans() + i*4,'-b')
		
		(anom_score, scores) = comb.get_scores(lsol, i, lats[i])
		plt.plot(range(LENS),scores.trans() + i*4,'-g')
	plt.show()

	(ssvm_err, ssvm_exm_err) = test.evaluate(pred)
	(ocsvm_err, ocsvm_exm_err) = test.evaluate(lats[EXMS_TRAIN:])
	print ssvm_err['fscore']
	print ocsvm_err['fscore']

	scores = []
	ascores = []
	for i in range(EXMS):
		(score, foo) = comb.get_scores(lsol, i, lats[i])
		if i>=EXMS_TRAIN:
			scores.append(score)
		ascores.append(score)


	(fpr,tpr,thres) = metric.roc_curve(labels, ascores)
	sauc = metric.auc(fpr, tpr)

 	print '#########################'
	print auc 
	print sauc

	plt.figure()
	#plt.plot(np.asarray(scores), np.asarray(ssvm_exm_err['fscore']),'.r')
	#plt.plot(np.asarray(scores), np.asarray(ocsvm_exm_err['fscore']),'.b')	

	#plt.plot(np.asarray(scores), np.asarray(ssvm_exm_err['fscore']),'.r')
	plt.plot(range(EXMS-EXMS_TRAIN), np.asarray(scores)/max(np.asarray(scores)),'.b')	
	plt.plot(range(EXMS-EXMS_TRAIN), np.asarray(oc_as[EXMS_TRAIN:])/max(np.asarray(oc_as[EXMS_TRAIN:])),'.r')	
	plt.plot(range(EXMS-EXMS_TRAIN), labels[EXMS_TRAIN:],'.g')	

	plt.show()

	print('finished')