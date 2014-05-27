import cvxopt as co
import numpy as np
import pylab as pl
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import scipy.io as io

from kernel import Kernel
from ssvm import SSVM
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
	X = remove_mean(X)
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)


def remove_mean(X):
	cnt = 0
	tst_mean = co.matrix(0.0, (1, 1))
	for i in range(len(X)):
		lens = len(X[i][:])
		cnt += lens
		tst_mean += co.matrix(1.0, (1, lens))*X[i].trans()
	tst_mean /= float(cnt)
	print tst_mean
	for i in range(len(X)):
		X[i][:] = X[i][:]-tst_mean
	cnt = 0
	tst_mean = co.matrix(0.0, (1, 1))
	for i in range(len(X)):
		lens = len(X[i][:])
		cnt += lens
		tst_mean += co.matrix(1.0, (1, lens))*X[i].trans()
	print tst_mean/float(cnt)
	return X


def experiment_anomaly_segmentation(train, test, comb, num_train, anom_prob, labels):

	# transductive train/pred for structured anomaly detection
	sad = StructuredOCSVM(comb, C=1.0/(num_train*0.4))
	(lsol, lats, thres) = sad.train_dc(max_iter=80)
	(cont, cont_exm) = test.evaluate(lats[num_train:])

	# train structured svm
	ssvm = SSVM(train)
	(sol, slacks) = ssvm.train()
	(vals, preds) = ssvm.apply(test)
	(base_cont, base_cont_exm) = test.evaluate(preds)

	return (cont, base_cont)


if __name__ == '__main__':
	LENS = 500
	EXMS = 400
	EXMS_TRAIN = 100
	ANOM_PROB = 0.05
	REPS = 5
	BLOCK_LEN = 300
	BLOCKS = [1]
	#BLOCKS = [1,2,5,10,50,100]

	# collected means
	conts = []
	conts_base = []
	for b in xrange(len(BLOCKS)):
		res = {}
		res_base = {}
		for r in xrange(REPS):
			(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, BLOCK_LEN, blocks=BLOCKS[b], anomaly_prob=ANOM_PROB)
			(cont, base_cont) = experiment_anomaly_segmentation(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
			for key in cont.keys():
				if r==0:
					res[key] = cont[key]/float(REPS)
					res_base[key] = base_cont[key]/float(REPS)
				else:
					res[key] += cont[key]/float(REPS)
					res_base[key] += base_cont[key]/float(REPS)

		conts.append((res['fscore'], res['precision'], res['sensitivity'], res['specificity']))
		conts_base.append((res_base['fscore'], res_base['precision'], res_base['sensitivity'], res_base['specificity']))

	print '####################'
	print('Mean    SAD={0} '.format(conts))
	print('Mean   SSVM={0} '.format(conts_base))
	print '####################'

	# store result as a file
	data = {}
	data['LENS'] = LENS
	data['EXMS'] = EXMS
	data['EXMS_TRAIN'] = EXMS_TRAIN
	data['ANOM_PROB'] = ANOM_PROB
	data['REPS'] = REPS
	data['BLOCKS'] = BLOCKS

	data['conts'] = conts
	data['conts_base'] = conts_base

	io.savemat('14_nips_toy_anno_03.mat',data)

	print('finished')