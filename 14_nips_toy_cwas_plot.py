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


def get_anom_model(num_exm, num_train, lens, blocks=1, anomaly_prob=0.15):
	print('Generating {0} sequences, {1} for training, each with {2} anomaly probability.'.format(num_exm, num_train, anomaly_prob))
	cnt = 0 
	X = [] 
	Y = []
	label = []
	for i in range(num_exm):
		(exm, lbl, marker) = ToyData.get_2state_anom_seq(lens, anom_prob=anomaly_prob, num_blocks=blocks)
		cnt += lens
		X.append(exm)
		Y.append(lbl)
		label.append(marker)
	X = remove_mean(X, 1)
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)

def get_model(num_exm, num_train, lens, feats, anomaly_prob=0.15):
	print('Generating {0} sequences, {1} for training, each with {2} anomaly probability.'.format(num_exm, num_train, anomaly_prob))
	cnt = 0 
	X = [] 
	Y = []
	label = []
	for i in range(num_exm):
		(exm, lbl, marker) = ToyData.get_2state_gaussian_seq(lens,dims=feats,anom_prob=anomaly_prob)
		cnt += lens
		X.append(exm)
		Y.append(lbl)
		label.append(marker)
	X = remove_mean(X, feats)
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


if __name__ == '__main__':
	LENS = 300
	EXMS = 250
	EXMS_TRAIN = 200
	ANOM_PROB = 0.1

	#(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, feats=2, anomaly_prob=ANOM_PROB)
	(train, test, comb, labels) = get_anom_model(EXMS, EXMS_TRAIN, LENS, blocks=10, anomaly_prob=ANOM_PROB)
	lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*0.5))
	(lsol, lats, thres) = lsvm.train_dc(max_iter=40)

	for i in range(EXMS):
		if (labels[i]==0):
	 		plt.plot(range(LENS),comb.X[i].trans() - 2,'-m')
	 		plt.plot(range(LENS),lats[i].trans() + 0,'-r')
	 		plt.plot(range(LENS),comb.y[i].trans() + 2,'-b')
		
	 		(anom_score, scores) = comb.get_scores(lsol, i, lats[i])
	 		plt.plot(range(LENS),scores.trans() + 4,'-g')
			break

	plt.show()

	print('finished')