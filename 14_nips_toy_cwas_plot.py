import cvxopt as co
import numpy as np
import pylab as pl
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import scipy.io as io
import timeit 

from kernel import Kernel
from ocsvm import OCSVM
from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA

from toydata import ToyData
from so_hmm import SOHMM


def get_anom_model(num_exm, num_train, lens, block_len, blocks=1, anomaly_prob=0.15):
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

if __name__ == '__main__':

	SHOW_EXPERIMENT_RESULT = False
	PLOT_TOY_RESULTS = False
	GEN_TOY_SEQS = False
	RUNTIME = False
	PLOT_RUNTIME = True


	if (SHOW_EXPERIMENT_RESULT==True):
		data = io.loadmat('../14_nips_pgm_11.mat')
		#data = io.loadmat('14_nips_pgm_03.mat')
		#data = io.loadmat('../14_nips_wind_11.mat')
		print data

		auc = data['auc'][0]
		base_auc = data['base_auc'][0]
		res = data['res']
		base_res = data['base_res']

		print auc
		reps = len(auc)

		# filter
		if 1==2:
			(foo, inds) = np.where([auc>0.75])
			print inds
			reps = len(inds)
			auc = auc[inds]
			base_auc = base_auc[inds]
			res = res[inds,:]
			base_res = base_res[inds,:]

		print reps
		mauc = sum(auc)/reps
		print mauc
		mbase_auc = sum(base_auc)/reps
		print mbase_auc

		vauc = np.sqrt(np.sum(co.mul(co.matrix(auc-mauc),co.matrix(auc-mauc)))/reps)
		vbase_auc = np.sqrt(np.sum(co.mul(co.matrix(base_auc-mbase_auc),co.matrix(base_auc-mbase_auc)))/reps)
		print vbase_auc


		m = np.sum(res,0)/reps
		print m
		bm = np.sum(base_res,0)/reps
		
		print (res[0,:])
		print np.sum([(res[i,:]-m)**2 for i in range(int(reps))],0)/reps

		std = np.sqrt(np.sum([(res[i,:]-m)**2 for i in range(int(reps)) ],0)/reps)
		bstd = np.sqrt(np.sum([(base_res[i,:]-bm)**2 for i in range(int(reps)) ],0)/reps)

		print('{0:1.2f}+/-{1:1.2f} & - & - & - & -'.format(mbase_auc, vbase_auc))	
		print('{0:1.2f}+/-{1:1.2f} & {2:1.2f}+/-{3:1.2f} & {4:1.2f}+/-{5:1.2f} & {6:1.2f}+/-{7:1.2f} & {8:1.2f}+/-{9:1.2f}'.format(mauc,vauc,m[0],std[0],m[1],std[1],m[2],std[2],m[3],std[3],))	
		print('- &  {0:1.2f}+/-{1:1.2f} & {2:1.2f}+/-{3:1.2f} & {4:1.2f}+/-{5:1.2f} & {6:1.2f}+/-{7:1.2f}'.format(bm[0],bstd[0],bm[1],bstd[1],bm[2],bstd[2],bm[3],bstd[3],))	


	if PLOT_TOY_RESULTS==True:
		data = io.loadmat('../14_nips_toy_anno_07.mat')
		print data
		inds=[0,1,2,3,5]
		blocks = data['BLOCKS'][0][inds]
		blocks = blocks[::-1]
		#blocks = range(6)
		print blocks


		conts = data['conts'][inds,:]
		var =  np.sqrt(data['var'][inds,:])
		conts_base = data['conts_base'][inds,:]
		var_base =  np.sqrt(data['var_base'][inds,:])

		title = ['F-score','Precision','Sensitivity','Specificity']
		for i in range(4):
			plt.subplot(2,2,i+1)
			plt.errorbar(blocks, conts_base[:,i], yerr=var_base[:,i], fmt='-om',linewidth=3.0, aa=True)
			plt.errorbar(blocks, conts[:,i], yerr=var[:,i], fmt='-ob',linewidth=2.0, aa=True)
			plt.xscale('log')
			plt.title(title[i])
			plt.xlim([0,100])
			plt.ylim([0.0,1.09])
			plt.xticks([])
			if (i==3):
				plt.legend(['Structured Anomaly Detection', 'Structured SVM'],loc=3)

		plt.xlim([0,100])
		plt.ylim([0.0,1.09])
		plt.show()

		data = io.loadmat('../14_nips_toy_ad_09.mat')
		print data

		blocks = data['BLOCKS'][0][::-1]
		#blocks = blocks[0:len(blocks)-1]

		auc = data['mauc'][0]
		vauc = np.sqrt(data['vauc'][0])
		mbase_auc = data['mbase_auc'][0]
		vbase_auc = np.sqrt(data['vbase_auc'][0])
		mbayes_auc = data['mbayes_auc'][0]
		vbayes_auc = np.sqrt(data['vbayes_auc'][0])

		plt.errorbar(blocks, mbayes_auc, yerr=vbayes_auc, fmt='--or',aa=True,linewidth=4.0)
		plt.errorbar(blocks, auc, yerr=vauc, fmt='-ob',aa=True,linewidth=2.0)
		plt.errorbar(blocks, mbase_auc, yerr=vbase_auc, fmt='-og',aa=True,linewidth=2.0)

		plt.legend(['Bayes Classifier','Structured Anomaly Detection', 'Vanilla One-class SVM'],loc=3)
		plt.xscale('log')
		plt.xlim([0,100])
		plt.ylim([0.0,1.09])
		plt.xticks([])
		plt.show()


	if GEN_TOY_SEQS==True:
		LENS = 600
		EXMS = 400
		EXMS_TRAIN = 100
		BLOCK_LEN = 100
		ANOM_PROB = 0.1

		(train, test, comb, labels) = get_anom_model(EXMS, EXMS_TRAIN, LENS, BLOCK_LEN, blocks=100, anomaly_prob=ANOM_PROB)
		lsvm = StructuredOCSVM(comb, C=1.0/(EXMS*anomaly_prob))
		(lsol, lats, thres) = lsvm.train_dc(max_iter=40)
		(err, err_exm) = comb.evaluate(lats)
		print err
		plt.figure(1)
		for i in range(EXMS):
			if (labels[i]==1):
				plt.plot(range(LENS),comb.X[i].trans()*0.2 - 2,'-m')
				plt.plot(range(LENS),lats[i].trans() + 0,'-r')
				plt.plot(range(LENS),comb.y[i].trans() + 2,'-b')
			
				(anom_score, scores) = comb.get_scores(lsol, i, lats[i])
				plt.plot(range(LENS),scores.trans() + 4,'-g')
				print anom_score
				break

		plt.show()


	if RUNTIME==True:
		LENS = 300
		EXMS = 601
		REPS = 10
		BLOCK_LEN = 100
		ANOM_PROB = 0.25
		TRAIN = [50,100,200,400,600]
		
		#TRAIN = [600]
		#REPS = 1

		times1 = co.matrix(0.0, (REPS,len(TRAIN)))
		times2 = co.matrix(0.0, (REPS,len(TRAIN)))
		times3 = co.matrix(0.0, (REPS,len(TRAIN)))
		for r in range(REPS):
			for n in range(len(TRAIN)):
				num = TRAIN[n]
				print num
				(train, test, comb, labels) = get_anom_model(EXMS, num, LENS, BLOCK_LEN, blocks=1, anomaly_prob=ANOM_PROB)
				phi = calc_feature_vecs(train.X)
				kern = Kernel.get_kernel(phi, phi)
				
				tic=timeit.default_timer()
				lsvm = StructuredOCSVM(train, C=1.0/(num*ANOM_PROB))
				(lsol, lats, thres) = lsvm.train_dc(max_iter=40)
				toc=timeit.default_timer()
				times1[r,n] = toc-tic
				
				tic=timeit.default_timer()
				ssvm = SSVM(train, C=10.0)
				(lsol,slacks) = ssvm.train()
				toc=timeit.default_timer()
				times2[r,n] = toc-tic

				tic=timeit.default_timer()
				ocsvm = OCSVM(kern, C=1.0/(num*ANOM_PROB))
				ocsvm.train_dual()
				toc=timeit.default_timer()
				times3[r,n] = toc-tic

		print times1
		print times2
		print times3
		data = {}
		data['times1'] = times1
		data['times2'] = times2
		data['times3'] = times3
		data['REPS'] = REPS
		data['TRAIN'] = TRAIN
		io.savemat('runtimes_clean_00.mat', data)


	if PLOT_RUNTIME==True:
		data = io.loadmat('runtimes_clean_00.mat')

		blocks = data['TRAIN']
		reps = data['REPS']
		t1 = data['times1']
		t2 = data['times2']
		t3 = data['times3']


		m1 = np.sum(t1,0)/float(reps)
		std1 = np.sqrt(np.sum([(t1[i,:]-m1)**2 for i in range(int(reps)) ],0)/float(reps))

		m2 = np.sum(t2,0)/float(reps)
		std2 = np.sqrt(np.sum([(t2[i,:]-m2)**2 for i in range(int(reps)) ],0)/float(reps))

		m3 = np.sum(t3,0)/float(reps)
		std3 = np.sqrt(np.sum([(t3[i,:]-m3)**2 for i in range(int(reps)) ],0)/float(reps))
		print m3

		plt.errorbar(blocks, m1, yerr=std1, fmt='--ob',aa=True,linewidth=2.0)
		plt.errorbar(blocks, m2, yerr=std2, fmt='-om',aa=True,linewidth=2.0)
		plt.errorbar(blocks, m3, yerr=std3, fmt='-og',aa=True,linewidth=2.0)
		plt.ylim([-4.0,55.0])

		plt.legend(['Structured Anomaly Detection', 'SSVM', 'Vanilla One-class SVM'],loc=0)
		plt.show()

	print('finished')