import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as io
import sklearn.metrics as metric
import csv

from kernel import Kernel
from ocsvm import OCSVM

from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA
from toydata import ToyData

from so_hmm import SOHMM


def smooth(x,window_len=4,window='blackman'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def normalize_sequence_data(X, dims=1):
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
            X[i][d,:] /= max_val[d]

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


def load_data(num_exms, path, fname, inds, label):
    LEN = 800
    DIMS = 5
    # training data
    trainX = []
    trainY = []
    start_symbs = []
    stop_symbs = []
    phi_list = []
    marker = []
    maxvals = co.matrix(0.0, (DIMS, 1))
    for i in xrange(num_exms):
        # load file 
        phi_i = co.matrix(0.0, (1, DIMS))
        lbl = co.matrix(0, (1,LEN))
        exm = co.matrix(0.0, (DIMS, LEN))
        with open('{0}{1}{2:03d}.csv'.format(path, fname, inds[i]+1)) as f:
            reader = csv.reader(f)
            idx = 0
            cdim = 0
            for row in reader:
                if idx==1:
                    for t in xrange(len(row)-1):
                        lbl[t] = int(row[t+1])-1
                if idx==3 or idx==5 or idx>3:
                    for t in xrange(len(row)-1):
                        exm[cdim, t] = float(row[t+1])
                        phi_i[cdim] += float(row[t+1])
                        if maxvals[cdim]<abs(float(row[t+1])):
                            maxvals[cdim] = float(row[t+1])
                    cdim += 1
                idx += 1
        marker.append(label)
        phi_list.append(phi_i)
        trainX.append(exm)
        trainY.append(lbl)
        phi_list[i] = phi_i
    return (trainX, trainY, phi_list, marker)

def build_histograms(data, phi, num_train, bins=2, ord=2):
    # first num_train phis are used for estimating
    # histogram boundaries.
    N = len(data)
    (F, LEN) = data[0].size
    print('(a) normalize features...')
    phi = normalize_features(phi, ord=ord)

    max_phi = np.max(phi[:,:num_train])
    min_phi = np.min(phi[:,:num_train])
    print("Build histograms with {0} bins.".format(bins))
    print (max_phi, min_phi)
    thres = np.linspace(min_phi, max_phi+1e-8, bins+1)
    print (max_phi, min_phi)

    hist = co.matrix(0.0, (F*bins, 1))
    phi_hist = co.matrix(0.0, (F*bins, N))
    for i in xrange(N):
        for f in xrange(F):
            phi_hist[0 + f*bins,i] = np.where(np.array(data[i][f,:])<thres[0])[0].size
            for b in range(1,bins-1):
                cnt = np.where((np.array(data[i][f,:])>=thres[b]) & (np.array(data[i][f,:])<thres[b+1]))[0].size
                phi_hist[b + f*bins,i] = float(cnt)
            phi_hist[bins-1 + f*bins,i] = np.where(np.array(data[i][f,:])>=thres[bins-1])[0].size
        phi_hist[:,i] /= np.linalg.norm(phi_hist[:,i], ord=ord)
        hist += phi_hist[:,i]/float(N)
    print('Histogram:')
    print hist.trans()
    return phi_hist

def normalize_features(phi, ord=1):
    phi_norm = co.matrix(phi)
    for i in range(phi.size[1]):
        phi_norm[:,i] /= np.linalg.norm(phi_norm[:,i], ord=ord)
    return phi_norm


def perf_ocsvm(phi, marker, train, test, anom_prob, ord=1):
    #phi = phi[phi_inds.tolist(),:]
    print('(a) normalize features...')
    phi = normalize_features(phi, ord=ord)
    print('(b) Build kernel...')
    kern = Kernel.get_kernel(phi, phi)
    print('(c) Train OCSVM...')
    ocsvm = OCSVM(kern[train,train], C=1.0/(float(len(train))*(1.0-anom_prob)))
    ocsvm.train_dual()
    print('(d) Apply OCSVM...')
    (oc_as, foo) = ocsvm.apply_dual(kern[test, train[ocsvm.get_support_dual()]])
    (fpr, tpr, thres) = metric.roc_curve(co.matrix(marker)[test], oc_as)
    auc = metric.auc(fpr, tpr)
    print('(e) Return AUC={0}...'.format(auc))
    return auc


def perf_sad(test_inds, marker, train, test, anom_prob):
    # SAD annotation
    print('(a) Setup SAD...')
    lsvm = StructuredOCSVM(train, C=1.0/(train.samples*(1.0-anom_prob)), norm_ord=2)
    print('(b) Train SAD...')
    (lsol, lats, thres) = lsvm.train_dc(max_iter=100)
    print('(c) Evaluate SAD...')
    (scores, lats) = lsvm.apply(test)
    (err, err_exm) = test.evaluate(lats)
    res = (err['fscore'], err['precision'], err['sensitivity'], err['specificity'])
    (fpr, tpr, thres) = metric.roc_curve(co.matrix(marker)[test_inds], scores)
    auc = metric.auc(fpr, tpr)
    print('(d) Return AUC={0}...'.format(auc))
    print res
    return auc, res


def perf_ssvm(test_inds, marker, train, test):
    # SAD annotation
    print('(a) Setup SSVM...')
    ssvm = SSVM(train, C=10.0)
    print('(b) Train SSVM...')
    (lsol,slacks) = ssvm.train()
    print('(c) Evaluate SSVM...')
    (scores, lats) = ssvm.apply(test)
    (err, err_exm) = test.evaluate(lats)
    res = (err['fscore'], err['precision'], err['sensitivity'], err['specificity'])
    (fpr, tpr, thres) = metric.roc_curve(co.matrix(marker)[test_inds], -scores)
    auc = metric.auc(fpr, tpr)
    print('(d) Return AUC={0}...'.format(auc))
    print res
    return auc, res



if __name__ == '__main__':
    # load data file
    directory = '/home/nicococo/Code/wind/'
    directory = '/home/nico/Data/wind/'

    out_fname = '15_icml_wind_c'   
    
    DIMS = 5
    EXMS_ANOM = 200
    EXMS_NON = 200
    REPS = 20
    BLOCKS = [5, 10, 20, 30, 40, 60]
    #BLOCKS = [30]

    for b in BLOCKS:

        NUM_TRAIN_ANOM = b
        NUM_TRAIN_NON = 200-NUM_TRAIN_ANOM
        
        NUM_TEST_ANOM = 200-NUM_TRAIN_ANOM
        NUM_TEST_NON = 200-NUM_TRAIN_NON

        NUM_COMB_ANOM = NUM_TRAIN_ANOM+NUM_TEST_ANOM
        NUM_COMB_NON = NUM_TRAIN_NON+NUM_TEST_NON

        anom_prob = float(NUM_TRAIN_ANOM) / float(NUM_TRAIN_ANOM+NUM_TRAIN_NON)
        print('Anomaly probability is {0}.'.format(anom_prob))

        all_auc = {}
        all_res = {}
        for r in xrange(REPS):
            # shuffle genes and intergenics
            anom_inds = np.random.permutation(EXMS_ANOM)
            non_inds = np.random.permutation(EXMS_NON)

            # load genes and intergenic examples
            (combX, combY, phi_list, marker) = load_data(NUM_COMB_ANOM, directory, 'winddata_A15_only_', anom_inds, 0)
            (X, Y, phis, lbls) = load_data(NUM_COMB_NON, directory, 'winddata_C10_only_', non_inds, 1)
            combX.extend(X)
            combY.extend(Y)
            phi_list.extend(phis)
            marker.extend(lbls)
            EXMS = len(combY)
            combX = normalize_sequence_data(combX, DIMS)

            total_len = 0
            for i in range(EXMS):
                total_len += len(combY[i])
            print('---> Total length = {0}.'.format(total_len))

            trainX = combX[0:NUM_TRAIN_ANOM]
            trainX.extend(X[0:NUM_TRAIN_NON])
            trainY = combY[0:NUM_TRAIN_ANOM]
            trainY.extend(Y[0:NUM_TRAIN_NON])

            testX = combX[NUM_TRAIN_ANOM:NUM_COMB_ANOM]
            testX.extend(X[NUM_TRAIN_NON:NUM_COMB_NON])
            testY = combY[NUM_TRAIN_ANOM:NUM_COMB_ANOM]
            testY.extend(Y[NUM_TRAIN_NON:NUM_COMB_NON])

            train = SOHMM(trainX, trainY, num_states=2)
            test = SOHMM(testX, testY, num_states=2)
            comb = SOHMM(combX, combY, num_states=2)

            inds_train = co.matrix(range(NUM_TRAIN_ANOM) + range(NUM_COMB_ANOM, NUM_COMB_ANOM+NUM_TRAIN_NON))
            inds_test = co.matrix(range(NUM_TRAIN_ANOM,NUM_COMB_ANOM) + range(NUM_COMB_ANOM+NUM_TRAIN_NON, NUM_COMB_ANOM+NUM_COMB_NON))

            # init result cache
            if not all_auc.has_key('SSVM'):
                # collect aucs
                all_auc['OcSvm (Hist 4)'] = []
                all_auc['OcSvm (Hist 8)'] = []
                all_auc['OcSvm (Hist 16)'] = []
                all_auc['SSVM'] = []
                all_auc['HMAD'] = []
                # collect fscores,..
                all_res['SSVM'] = []
                all_res['HMAD'] = []

            # structured output svm
            #(auc, res) = perf_ssvm(inds_test, marker, train, test)
            #all_auc['SSVM'].append(auc)
            #all_res['SSVM'].append(res)

            num_train = NUM_TRAIN_ANOM+NUM_TRAIN_NON
            phis = co.matrix(phi_list).trans()
            phis1 = build_histograms(comb.X, phis, num_train, bins=4, ord=2)
            phis = co.matrix(phi_list).trans()
            phis2 = build_histograms(comb.X, phis, num_train, bins=8, ord=2)
            phis = co.matrix(phi_list).trans()
            phis4 = build_histograms(comb.X, phis, num_train, bins=16, ord=2)

            # spectrum kernel oc-svms
            auc = perf_ocsvm(phis1, marker, inds_train, inds_test, anom_prob)
            all_auc['OcSvm (Hist 4)'].append(auc)
            auc = perf_ocsvm(phis2, marker, inds_train, inds_test, anom_prob)
            all_auc['OcSvm (Hist 8)'].append(auc)
            auc = perf_ocsvm(phis4, marker, inds_train, inds_test, anom_prob)
            all_auc['OcSvm (Hist 16)'].append(auc)

            (auc, res) = perf_sad(inds_test, marker, train, test, anom_prob)
            all_auc['HMAD'].append(auc)
            all_res['HMAD'].append(res)


        print '##############################################'
        print  out_fname
        print '##############################################'
        print NUM_COMB_ANOM
        print NUM_COMB_NON
        print '##############################################'
        print total_len 
        print anom_prob
        print '##############################################'
        print all_auc
        print '##############################################'
        print all_res
        print '##############################################'
        # store result as a file
        data = {}
        data['auc'] = all_auc
        data['res'] = all_res
        data['anom_frac'] = anom_prob

        outfile = '{0}{1:1.1f}'.format(out_fname,anom_prob*100.0)
        io.savemat(outfile, data)

    print('finished')