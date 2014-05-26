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


def get_model(num_exm, num_train, lens, blocks=1, anomaly_prob=0.15):
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
		X[i][0,:] -= mean
	return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)


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
	LENS = 500
	EXMS = 400
	EXMS_TRAIN = 100
	ANOM_PROB = 0.05
	REPS = 20
	BLOCKS = [1,2,5,10,50,100]

	# collected means
	conts = []
	conts_base = []
	var_conts = []
	var_conts_base = []
	for b in xrange(len(BLOCKS)):
		res = {}
		res_base = {}
		var_res = []
		var_res_base = []
		for r in xrange(REPS):
			(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, blocks=BLOCKS[b], anomaly_prob=ANOM_PROB)
			(cont, base_cont) = experiment_anomaly_segmentation(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
			var_res.append(cont)
			var_res_base.append(base_cont)
			for key in cont.keys():
				if r==0:
					res[key] = cont[key]/float(REPS)
					res_base[key] = base_cont[key]/float(REPS)
				else:
					res[key] += cont[key]/float(REPS)
					res_base[key] += base_cont[key]/float(REPS)
		
		var = {}
		var_base = {}
		for key in res.keys():
			var[key] = sum([(var_res[i][key]-res[key])**2 for i in xrange(REPS)])/float(REPS)
			var_base[key] = sum([(var_res_base[i][key]-res_base[key])**2 for i in xrange(REPS)])/float(REPS)

		conts.append(res)
		conts_base.append(res_base)
		var_conts.append(var)
		var_conts_base.append(var_base)

	print '####################'
	print('Mean    SAD={0} '.format(conts))
	print('Mean   SSVM={0} '.format(conts_base))
	print '####################'
	print('Variance    SAD={0} '.format(var_conts))
	print('Variance   SSVM={0} '.format(var_conts_base))
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
	data['var_conts'] = var_conts
	data['var_conts_base'] = var_conts_base

	io.savemat('14_nips_toy_anno_03.mat',data)

	print('finished')