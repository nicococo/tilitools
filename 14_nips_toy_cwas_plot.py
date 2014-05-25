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


def get_model2(num_exm, num_train, lens, blocks=1, anomaly_prob=0.15):
	print('Generating {0} sequences, {1} for training, each with {2} anomaly probability.'.format(num_exm, num_train, anomaly_prob))
	mean = 0.0
	cnt = 0 
	X = [] 
	Y = []
	label = []
	for i in range(num_exm):
		(exm, lbl, marker) = ToyData.get_2state_anom_seq(lens, anom_prob=anomaly_prob, num_blocks=blocks)
		mean += co.matrix(1.0, (1, lens))*exm.trans()
		cnt += lens
		X.append(exm)
		Y.append(lbl)
		label.append(marker)
	mean = mean / float(cnt)
	for i in range(num_exm):
		X[i] -= mean
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)

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


if __name__ == '__main__':
	LENS = 300
	EXMS = 250
	EXMS_TRAIN = 200
	ANOM_PROB = 0.1

	#(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, feats=2, anomaly_prob=ANOM_PROB)
	(train, test, comb, labels) = get_model2(EXMS, EXMS_TRAIN, LENS, blocks=10, anomaly_prob=ANOM_PROB)
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