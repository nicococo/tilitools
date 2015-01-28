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
	lblcnt = co.matrix(0.0,(1,lens))
	for i in range(num_exm):
		(exm, lbl, marker) = ToyData.get_2state_anom_seq(lens, block_len, anom_prob=anomaly_prob, num_blocks=blocks)
		cnt += lens
		X.append(exm)
		Y.append(lbl)
		label.append(marker)
		# some lbl statistics
		if i<num_train:
			lblcnt += lbl
	X = remove_mean(X,1)
	#plt.figure(1)
	#plt.plot(range(lens), lblcnt.trans(), '-r')
	#plt.show()
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
	
	max_val = co.matrix(-1e10, (1, dims))
	for i in range(len(X)):
		for d in range(dims):
			X[i][d,:] = X[i][d,:]-tst_mean[d]
			foo = np.max(np.abs(X[i][d,:]))
			max_val[d] = np.max([max_val[d], foo])
	
	print max_val
	for i in range(len(X)):
		for d in range(dims):
			#X[i][d,:] *= -1
			X[i][d,:] /= max_val[d]
			#X[i][d,:] /= 600.0
			#X[i][d,:] /= np.linalg.norm(X[i][d,:])

	cnt = 0
	max_val = co.matrix(-1e10, (1, dims))
	tst_mean = co.matrix(0.0, (1, dims))
	for i in range(len(X)):
		lens = len(X[i][0,:])
		cnt += lens
		tst_mean += co.matrix(1.0, (1, lens))*X[i].trans()
		for d in range(dims):
			foo = np.max(np.abs(X[i][d,:]))
			max_val[d] = np.max([max_val[d], foo])
	print tst_mean/float(cnt)
	print max_val
	return X

def calc_feature_vecs(data):
	# ASSUME that all sequences have the same length!
	N = len(data)
	(F, LEN) = data[0].size
	phi = co.matrix(0.0, (F*LEN, N))
	for i in xrange(N):
		for f in xrange(F):
			phi[(f*LEN):(f*LEN)+LEN,i] = data[i][f,:].trans()
		#norm = np.linalg.norm(phi[:,i],2)
		#print norm
		#phi[:,i] /= norm

	# histogram
	max_phi = np.max(phi)
	min_phi = np.min(phi)

	BINS = 2
	thres = np.linspace(min_phi, max_phi+1e-8, BINS+1)

	print (max_phi, min_phi)
	phi_hist4 = co.matrix(0.0, (F*BINS, N))
	for i in xrange(N):
		for f in xrange(F):
			for b in range(BINS):
				cnt = np.where((np.array(data[i][f,:])>=thres[b]) & (np.array(data[i][f,:])<thres[b+1]))[0].size
				#print cnt
				phi_hist4[b + f*BINS,i] = float(cnt)
		phi_hist4[:,i] /= np.linalg.norm(phi_hist4[:,i])
	print phi_hist4

	BINS = 8
	thres = np.linspace(min_phi, max_phi+1e-8, BINS+1)
	#thres = np.array([-10., 1., 10.])
	phi_hist8 = co.matrix(0.0, (F*BINS, N))
	for i in xrange(N):
		for f in xrange(F):
			for b in range(BINS):
				cnt = np.where((np.array(data[i][f,:])>=thres[b]) & (np.array(data[i][f,:])<thres[b+1]))[0].size
				#print cnt
				phi_hist8[b + f*BINS,i] = float(cnt)
		phi_hist8[:,i] /= np.linalg.norm(phi_hist8[:,i])
		#phi_hist8[:,i] = (1.0-phi_hist8[:,i])
	print phi_hist8

	return phi, phi_hist4, phi_hist8  


def experiment_anomaly_detection(train, test, comb, num_train, anom_prob, labels):
	phi, phi_hist4, phi_hist8 = calc_feature_vecs(comb.X)
	print phi.size
	bayes_auc = 0.0
	auc = 0.0
	base_auc = 0.0
	base_auc1 = 0.0
	base_auc2 = 0.0
	base_auc3 = 0.0
	base_auc4 = 0.0
	base_auc5 = 0.0
	base_auc6 = 0.0
	base_hist_auc1 = 0.0
	base_hist_auc2 = 0.0

	# bayes classifier
	(DIMS, N) = phi.size
	w_bayes = co.matrix(-1.0, (DIMS, 1))
	pred = w_bayes.trans()*phi[:,num_train:]
	(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred.trans())
	bayes_auc = metric.auc(fpr, tpr)

	ALL = True
	if ALL:
		# oc-svm without pre-processing
		# normalize data l1
		phil1 = co.matrix(phi)
		for idx in range(phil1.size[1]):
			phil1[:,idx] /= np.linalg.norm(phil1[:,idx],ord=1)

		kern = Kernel.get_kernel(phil1[:,0:num_train], phil1[:,0:num_train])
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		ocsvm.train_dual()
		kern = Kernel.get_kernel(phil1, phil1)
		(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
		(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
		base_auc = metric.auc(fpr, tpr)

		# train histogram 4 one-class svm
		kern = Kernel.get_kernel(phi_hist4[:,0:num_train], phi_hist4[:,0:num_train])
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		ocsvm.train_dual()
		kern = Kernel.get_kernel(phi_hist4, phi_hist4)
		(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
		(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
		base_hist_auc1 = metric.auc(fpr, tpr)

		# train histogram 8 one-class svm
		kern = Kernel.get_kernel(phi_hist8[:,0:num_train], phi_hist8[:,0:num_train])
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		ocsvm.train_dual()
		kern = Kernel.get_kernel(phi_hist8, phi_hist8)
		(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
		(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
		base_hist_auc2 = metric.auc(fpr, tpr)

		# plt.figure(1)
		# lbls = np.array(labels[num_train:], dtype='int')
		# inds = np.where(lbls==1)[0].tolist()
		# plt.plot(range(len(oc_as)), np.array(oc_as), '.r')
		# for i in inds:
		# 	plt.plot(i, np.array(oc_as[i]), 'ob')
		#plt.show()

		# normalize data
		for idx in range(phi.size[1]):
			phi[:,idx] /= np.linalg.norm(phi[:,idx])

		# train one-class svm
		kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train])
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		msg = ocsvm.train_dual()
		if msg==OCSVM.MSG_ERROR:
			base_auc1 = 0.5
		else:
			kern = Kernel.get_kernel(phi, phi)
			(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
			(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
			base_auc1 = metric.auc(fpr, tpr)

		# train one-class svm RBF
		kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=0.1)
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		msg = ocsvm.train_dual()
		if msg==OCSVM.MSG_ERROR:
			base_auc2 = 0.5
		else:
			kern = Kernel.get_kernel(phi, phi, type='rbf', param=0.1)
			(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
			(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
			base_auc2 = metric.auc(fpr, tpr)

		kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=1.0)
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		msg = ocsvm.train_dual()
		if msg==OCSVM.MSG_ERROR:
			base_auc3 = 0.5
		else:
			kern = Kernel.get_kernel(phi, phi, type='rbf', param=1.0)
			(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
			(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
			base_auc3 = metric.auc(fpr, tpr)

		kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=2.0)
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		msg = ocsvm.train_dual()
		if msg==OCSVM.MSG_ERROR:
			base_auc4 = 0.5
		else:
			kern = Kernel.get_kernel(phi, phi, type='rbf', param=2.0)
			(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
			(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
			base_auc4 = metric.auc(fpr, tpr)

		kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=4.0)
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		msg = ocsvm.train_dual()
		if msg==OCSVM.MSG_ERROR:
			base_auc5 = 0.5
		else:
			kern = Kernel.get_kernel(phi, phi, type='rbf', param=4.0)
			(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
			(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
			base_auc5 = metric.auc(fpr, tpr)

		kern = Kernel.get_kernel(phi[:,0:num_train], phi[:,0:num_train], type='rbf', param=10.0)
		ocsvm = OCSVM(kern, C=1.0/(num_train*anom_prob))
		msg = ocsvm.train_dual()
		if msg==OCSVM.MSG_ERROR:
			base_auc6 = 0.5
		else:
			kern = Kernel.get_kernel(phi, phi, type='rbf', param=10.0)
			(oc_as, foo) = ocsvm.apply_dual(kern[num_train:,ocsvm.get_support_dual()])
			(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], oc_as)
			base_auc6 = metric.auc(fpr, tpr)
		print base_auc6

	if ALL:
		# train structured anomaly detection
		sad = StructuredOCSVM(train, C=1.0/(num_train*anom_prob))
		(lsol, lats, thres) = sad.train_dc(max_iter=60,zero_shot=False)
		(pred_vals, pred_lats) = sad.apply(test)	
		(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred_vals)
		auc = metric.auc(fpr, tpr)

		plt.figure(2)
		(pred_vals, pred_lats) = sad.apply(train)
		oc_as = pred_vals
		lbls = np.array(labels[:num_train], dtype='int')
		inds = np.where(lbls==0)[0].tolist()
		svs = sad.svs_inds

		# plt.plot(range(len(oc_as)), np.array(oc_as), '.r')
		# for i in inds:
		# 	plt.plot(i, np.array(oc_as[i]), 'ob')
		# for i in svs:
		# 	plt.plot(i, np.array(oc_as[i]), 'r', marker='x', markersize=10)
		# plt.show()
	return (auc, base_auc, bayes_auc, base_auc1, base_auc2, base_auc3, base_auc4, base_auc5, base_auc6, base_hist_auc1, base_hist_auc2)


if __name__ == '__main__':
	LENS = 600
	EXMS = 800
	EXMS_TRAIN = 400
	ANOM_PROB = 0.1
	REPS = 10
	BLOCK_LEN = 100
	#BLOCKS = [1,100]
	BLOCKS = [1,2,5,10,20,40,60,80,100]
	#BLOCKS = [1]

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

	mbase_hist_auc1 = []
	mbase_hist_auc2 = []

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

	vbase_hist_auc1 = []
	vbase_hist_auc2 = []
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
		fmbase_hist_auc1 = 0.0
		fmbase_hist_auc2 = 0.0
		for r in xrange(REPS):
			(train, test, comb, labels) = get_model(EXMS, EXMS_TRAIN, LENS, BLOCK_LEN, blocks=BLOCKS[b], anomaly_prob=ANOM_PROB)
			(auc, base_auc, bayes_auc, base_auc1, base_auc2, base_auc3, base_auc4, base_auc5, base_auc6, base_hist_auc1, base_hist_auc2)  = experiment_anomaly_detection(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)
			aucs.append((auc, base_auc, bayes_auc, base_auc1, base_auc2, base_auc3, base_auc4, base_auc5, base_auc6, base_hist_auc1, base_hist_auc2))
			fmauc += auc
			fmbase_auc += base_auc
			fmbayes_auc += bayes_auc

			fmbase_auc1 += base_auc1
			fmbase_auc2 += base_auc2
			fmbase_auc3 += base_auc3
			fmbase_auc4 += base_auc4
			fmbase_auc5 += base_auc5
			fmbase_auc6 += base_auc6

			fmbase_hist_auc1 += base_hist_auc1
			fmbase_hist_auc2 += base_hist_auc2

		mauc.append(fmauc/float(REPS))
		mbase_auc.append(fmbase_auc/float(REPS))
		mbayes_auc.append(fmbayes_auc/float(REPS))

		mbase_auc1.append(fmbase_auc1/float(REPS))
		mbase_auc2.append(fmbase_auc2/float(REPS))
		mbase_auc3.append(fmbase_auc3/float(REPS))
		mbase_auc4.append(fmbase_auc4/float(REPS))
		mbase_auc5.append(fmbase_auc5/float(REPS))
		mbase_auc6.append(fmbase_auc6/float(REPS))

		mbase_hist_auc1.append(fmbase_hist_auc1/float(REPS))
		mbase_hist_auc2.append(fmbase_hist_auc2/float(REPS))

		vauc.append(sum([ (aucs[i][0]-mauc[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc.append(sum([ (aucs[i][1]-mbase_auc[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbayes_auc.append(sum([ (aucs[i][2]-mbayes_auc[b])**2 for i in xrange(REPS)]) / float(REPS))

		vbase_auc1.append(sum([ (aucs[i][3]-mbase_auc1[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc2.append(sum([ (aucs[i][4]-mbase_auc2[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc3.append(sum([ (aucs[i][5]-mbase_auc3[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc4.append(sum([ (aucs[i][6]-mbase_auc4[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc5.append(sum([ (aucs[i][7]-mbase_auc5[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_auc6.append(sum([ (aucs[i][8]-mbase_auc6[b])**2 for i in xrange(REPS)]) / float(REPS))

		vbase_hist_auc1.append(sum([ (aucs[i][9]-mbase_hist_auc1[b])**2 for i in xrange(REPS)]) / float(REPS))
		vbase_hist_auc2.append(sum([ (aucs[i][10]-mbase_hist_auc2[b])**2 for i in xrange(REPS)]) / float(REPS))


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
	print('Mean/Variance  OCSVM HIST 4={0} / {1}'.format(mbase_hist_auc1, vbase_hist_auc1))
	print('Mean/Variance  OCSVM HIST 8={0} / {1}'.format(mbase_hist_auc2, vbase_hist_auc2))
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

	data['mbase_hist_auc1'] = mbase_hist_auc1
	data['mbase_hist_auc2'] = mbase_hist_auc2
	data['vbase_hist_auc1'] = vbase_hist_auc1
	data['vbase_hist_auc2'] = vbase_hist_auc2

	io.savemat('15_icml_toy_ad_05.mat',data)

	print('finished')
