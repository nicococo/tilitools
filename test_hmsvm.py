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


def experiment_anomaly_detection(train, test, comb, num_train, anom_prob, labels):
	# train one-class svm
	phi = calc_feature_vecs(comb.X)
	kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train])
	ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
	ocsvm.train_dual()
	kern = Kernel.get_kernel(phi, phi)
	(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
	base_auc = metric.auc(fpr, tpr)

	# train structured anomaly detection
	sad = StructuredOCSVM(train, C=1.0/(num_train*0.5))
	(lsol, lats, thres) = sad.train_dc(max_iter=40)
	(pred_vals, pred_lats) = sad.apply(test)
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred_vals)
	auc = metric.auc(fpr, tpr)

	return (auc, base_auc)


def experiment_anomaly_segmentation(train, test, comb, num_train, anom_prob, labels):

	# transductive train/pred for structured anomaly detection
	sad = StructuredOCSVM(comb, C=1.0/(num_train*0.5))
	(lsol, lats, thres) = sad.train_dc(max_iter=40)
	(cont, cont_exm) = test.evaluate(lats[num_train:])

	# train structured svm
	ssvm = SSVM(train)
	(sol, slacks) = ssvm.train()
	(vals, preds) = ssvm.apply(test)
	(base_cont, base_cont_exm) = test.evaluate(preds)

	return (cont, base_cont)



if __name__ == '__main__':
	DIMS = 2
	LENS = 250
	EXMS = 300
	EXMS_TRAIN = 100
	ANOM_PROB = 0.15

	# generate data
	(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, DIMS, ANOM_PROB)

	(auc, base_auc) = experiment_anomaly_detection(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
	print '####################'
	print auc
	print base_auc

	(cont, base_cont) = experiment_anomaly_segmentation(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
	print '####################'
	print cont
	print base_cont

	# ssvm = SSVM(train)
	# (sol, slacks) = ssvm.train()

	# kern = Kernel.get_kernel(phi, phi)
	# ocsvm = OCSVM(kern, C=1.0/(EXMS*0.15))
	# ocsvm.train_dual()
	# (oc_as, foo) = ocsvm.apply_dual(kern[:,ocsvm.get_support_dual()])
	# (fpr,tpr,thres) = metric.roc_curve(labels, oc_as)
	# auc = metric.auc(fpr, tpr)
 # 	print '#########################'
	# print auc 


	# lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*0.5))
	# (lsol, lats, thres) = lsvm.train_dc(max_iter=40)

	# # test
	# (vals, pred) = ssvm.apply(test)

	# plt.figure()
	# for d in range(DIMS):
	# 	plt.plot(range(LENS),test.X[0][d,:].trans() + 5.0*d,'-r')
	
	# (anom_score, scores) = test.get_scores(lsol, 0, pred[0])
	# plt.plot(range(LENS),scores.trans() + 5.0,'-b')

	# plt.plot(range(LENS),pred[0].trans() - 5,'-g')
	# plt.plot(range(LENS),lats[EXMS_TRAIN].trans() - 7.5,'-r')
	# plt.plot(range(LENS),test.y[0].trans() - 10,'-b')
	# plt.show()

	# for i in range(20):
	# 	plt.plot(range(LENS),lats[i].trans() + i*4,'-r')
	# 	plt.plot(range(LENS),comb.y[i].trans() + i*4,'-b')
		
	# 	(anom_score, scores) = comb.get_scores(lsol, i, lats[i])
	# 	plt.plot(range(LENS),scores.trans() + i*4,'-g')
	# plt.show()

	# (ssvm_err, ssvm_exm_err) = test.evaluate(pred)
	# (ocsvm_err, ocsvm_exm_err) = test.evaluate(lats[EXMS_TRAIN:])
	# print ssvm_err['fscore']
	# print ocsvm_err['fscore']

	# scores = []
	# ascores = []
	# for i in range(EXMS):
	# 	(score, foo) = comb.get_scores(lsol, i, lats[i])
	# 	if i>=EXMS_TRAIN:
	# 		scores.append(score)
	# 	ascores.append(score)


	# (fpr,tpr,thres) = metric.roc_curve(labels, ascores)
	# sauc = metric.auc(fpr, tpr)

 # 	print '#########################'
	# print auc 
	# print sauc

	# plt.figure()
	# #plt.plot(np.asarray(scores), np.asarray(ssvm_exm_err['fscore']),'.r')
	# #plt.plot(np.asarray(scores), np.asarray(ocsvm_exm_err['fscore']),'.b')	

	# #plt.plot(np.asarray(scores), np.asarray(ssvm_exm_err['fscore']),'.r')
	# plt.plot(range(EXMS-EXMS_TRAIN), np.asarray(scores)/max(np.asarray(scores)),'.b')	
	# plt.plot(range(EXMS-EXMS_TRAIN), np.asarray(oc_as[EXMS_TRAIN:])/max(np.asarray(oc_as[EXMS_TRAIN:])),'.r')	
	# plt.plot(range(EXMS-EXMS_TRAIN), labels[EXMS_TRAIN:],'.g')	

	# plt.show()

	print('finished')