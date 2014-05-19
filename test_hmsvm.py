import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA

from toydata import ToyData
from so_hmm import SOHMM


def get_model(num_exm, num_train, lens, feats):
	mean = 0.0
	cnt = 0 
	X = []
	Y = []
	for i in range(num_exm):
		(exm,lbl) = ToyData.get_2state_gaussian_seq(lens,dims=feats,anom_prob=0.15)
		#if i<4:
		#	(exm,lbl) = ToyData.get_2state_gaussian_seq(LENS,dims=2,means1=[1,-3],means2=[3,7],vars1=[1,1],vars2=[1,1])
		mean += co.matrix(1.0, (1, lens))*exm.trans()
		cnt += lens
		X.append(exm)
		Y.append(lbl)
	mean = mean / float(cnt)
	for i in range(num_exm):
		#if (i<10):
		#	pos = int(np.single(co.uniform(1))*float(LENS)*0.8 + 4.0)
		#	print pos
		#	trainX[i][1,pos] = 100.0
		for d in range(feats):
			X[i][d,:] = X[i][d,:]-mean[d]
	# return train, test, combined
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y))



if __name__ == '__main__':
	DIMS = 2
	LENS = 250
	EXMS = 200
	EXMS_TRAIN = 100

	# training data
	(train, test, comb) = get_model(EXMS, EXMS_TRAIN, LENS, DIMS)

	ssvm = SSVM(train)
	(sol, slacks) = ssvm.train()

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

	(ssvm_err, ssvm_exm_err, base) = test.evaluate(pred)
	(ocsvm_err, ocsvm_exm_err, base) = test.evaluate(lats[EXMS_TRAIN:])
	print ssvm_err
	print ocsvm_err
	print base

	scores = []
	for i in range(EXMS-EXMS_TRAIN):
		(score, foo) = test.get_scores(lsol, i, lats[i+EXMS_TRAIN])
		scores.append(score)

	plt.figure()
	plt.plot(np.asarray(scores), np.asarray(ssvm_exm_err),'.r')
	plt.plot(np.asarray(scores), np.asarray(ocsvm_exm_err),'.b')
	plt.show()

	print('finished')