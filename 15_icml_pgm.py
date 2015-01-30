import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as io
import sklearn.metrics as metric
import sys
import argparse

from kernel import Kernel
from ocsvm import OCSVM

from ssvm import SSVM
from latentsvdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA
from toydata import ToyData

from so_pgm import SOPGM
from so_hmm import SOHMM


def add_intergenic(num_exm, signal, label, region_start, region_end, exm_lens):
    trainX = []
    trainY = []
    phi1 = []
    phi2 = []
    phi3 = []
    DIMS = 4**3
    for i in range(num_exm):
        inds = range(region_start+i*exm_lens,region_start+(i+1)*exm_lens)
        lens = len(inds)
        print('Index {0}: #{1}'.format(i,lens))
        phi1_i = co.matrix(0.0, (1, DIMS))
        phi2_i = co.matrix(0.0, (1, DIMS + DIMS*DIMS))
        phi3_i = co.matrix(0.0, (1, DIMS + DIMS*DIMS + DIMS*DIMS*DIMS))
        error = 0
        mod = 1
        lbl = co.matrix(0, (1, lens))
        exm = co.matrix(-1.0, (DIMS, lens))
        for t in range(lens):
            exm[ int(np.int32(signal[0,inds[t]])), t] = 1.0
            # spectrum kernel entry
            phi1_i[int(np.int32(signal[0,inds[t]]))] +=1.0
            phi2_i[int(np.int32(signal[0,inds[t]]))] +=1.0
            phi3_i[int(np.int32(signal[0,inds[t]]))] +=1.0
            if t<lens-1:
                a_mere = int(np.int32(signal[0,inds[t]]))
                b_mere = int(np.int32(signal[0,inds[t+1]]))
                phi2_i[DIMS + a_mere*DIMS+b_mere] += 1.0
                phi3_i[DIMS + a_mere*DIMS+b_mere] += 1.0
            if t<lens-2:
                a_mere = int(np.int32(signal[0,inds[t]]))
                b_mere = int(np.int32(signal[0,inds[t+1]]))
                c_mere = int(np.int32(signal[0,inds[t+2]]))
                phi3_i[DIMS + DIMS*DIMS + a_mere*DIMS*DIMS+b_mere*DIMS+c_mere] += 1.0

            val = max(0, label[0,inds[t]])
            error += val
            lbl[t] = int(val)

        phi1.append(phi1_i)
        phi2.append(phi2_i)
        phi3.append(phi3_i)
        if (error>0):
            print 'ERROR loading integenic regions: gene found!'
        trainX.append(exm)
        trainY.append(lbl)
    return (trainX, trainY, phi1, phi2, phi3)


def find_intergenic_regions(labels, min_gene_dist=20):
    (foo, N) = labels.shape
    total_len = 0
    cnt = 0
    ige_intervals = []
    start = 0
    stop = 0
    isCnt = True
    for t in xrange(N):
        if (isCnt==True and labels[0,t]>0):
            isCnt = False
            stop = t-1
            dist = stop-start
            if dist>2*min_gene_dist:
                ige_intervals.append((start+min_gene_dist, stop-min_gene_dist))
                cnt += 1
                total_len += dist-2*min_gene_dist
        if (isCnt==False and labels[0,t]<0):
            start = t
            isCnt = True
    print('Found {0} intergenic regions with a total length of {1}.'.format(cnt, total_len))
    return ige_intervals


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


def load_genes(max_genes, signal, label, exm_id_intervals, min_lens=600, max_lens=800):
    DIMS = 4**3
    EXMS = len(exm_id_intervals[:,0])

    # training data
    trainX = []
    trainY = []
    start_symbs = []
    stop_symbs = []
    phi1_list = []
    phi2_list = []
    phi3_list = []
    marker = []
    for i in xrange(EXMS):
        # convert signal to binary feature array
        # convert labels to states
        #(foo,inds) = np.where([exm_id[0,:]==i])
        inds = range(exm_id_intervals[i,1]-1,exm_id_intervals[i,2])
        lens = len(inds)
        if lens>max_lens or lens<=min_lens:
            continue
        max_genes -= 1
        if max_genes<0:
            break
        print('Index {0}: #{1}'.format(i,lens))
        mod = 1
        lbl = co.matrix(0, (1, lens))
        exm = co.matrix(0.0, (DIMS, lens))
        phi1_i = co.matrix(0.0, (1, DIMS))
        phi2_i = co.matrix(0.0, (1, DIMS + DIMS*DIMS))
        phi3_i = co.matrix(0.0, (1, DIMS + DIMS*DIMS + DIMS*DIMS*DIMS))
        for t in range(lens):
            exm[ int(np.int32(signal[0,inds[t]])), t ] = 1.0
            # labels to states
            val = max(0, label[0,inds[t]])
            if val==0 or val==1: 
                mod=1
            if val==3:
                lbl[t] = int(val + mod)
                mod = (mod+1) % 3
            else:
                lbl[t] = int(val)
            # store start/stop symbols
            if (label[0,inds[t]]==1):
                start_symbs.append(signal[0,inds[t]])
            if (label[0,inds[t]]==2):
                stop_symbs.append(signal[0,inds[t]])
            # spectrum kernel entry
            phi1_i[int(np.int32(signal[0,inds[t]]))] +=1.0
            phi2_i[int(np.int32(signal[0,inds[t]]))] +=1.0
            phi3_i[int(np.int32(signal[0,inds[t]]))] +=1.0
            if t<lens-1:
                a_mere = int(np.int32(signal[0,inds[t]]))
                b_mere = int(np.int32(signal[0,inds[t+1]]))
                phi2_i[DIMS + a_mere*DIMS+b_mere] += 1.0
                phi3_i[DIMS + a_mere*DIMS+b_mere] += 1.0
            if t<lens-2:
                a_mere = int(np.int32(signal[0,inds[t]]))
                b_mere = int(np.int32(signal[0,inds[t+1]]))
                c_mere = int(np.int32(signal[0,inds[t+2]]))
                phi3_i[DIMS + DIMS*DIMS + a_mere*DIMS*DIMS+b_mere*DIMS+c_mere] += 1.0

        marker.append(0)
        phi1_list.append(phi1_i)
        phi2_list.append(phi2_i)
        phi3_list.append(phi3_i)
        trainX.append(exm)
        trainY.append(lbl)
    print '###################'
    print start_symbs
    print stop_symbs 
    print '###################'
    return (trainX, trainY, phi1_list, phi2_list, phi3_list, marker)


def load_intergenics(num_iges, signal, label, ige_intervals, min_lens=600, max_lens=800):
    # add intergenic examples
    marker = []
    trainX = []
    trainY = []
    phi1_list = []
    phi2_list = []
    phi3_list = []
    ige_cnt = 0
    IGE_EXMS = num_iges
    N = len(ige_intervals)
    for i in xrange(N):
        lens = ige_intervals[i][1]-ige_intervals[i][0]
        if lens>10000:
            IGE_LEN = np.int(np.single(co.uniform(1, a=min_lens, b=max_lens)))
            num_ige_exms = np.int(np.floor(float(lens)/float(IGE_LEN)))
            if (num_ige_exms > IGE_EXMS-ige_cnt):
                num_ige_exms = IGE_EXMS-ige_cnt
            ige_cnt += num_ige_exms
            (X, Y, phis1, phis2, phis3) = add_intergenic(num_ige_exms, signal, label, ige_intervals[i][0], ige_intervals[i][1], IGE_LEN)
            trainX.extend(X)
            trainY.extend(Y)
            phi1_list.extend(phis1)
            phi2_list.extend(phis2)
            phi3_list.extend(phis3)
            for j in range(num_ige_exms):
                marker.append(1)
        if ige_cnt>IGE_EXMS:
            break
    print('IGE examples {0}'.format(ige_cnt))
    return (trainX, trainY, phi1_list, phi2_list, phi3_list, marker)


def feature_selection(fname, top_k=32, num_exms=100):
    """ Simple feature selection technique:
        (a) choose a different but related organism (e.g. e.fergusonii for e.coli)
        (b) load a certain number of genes and intergenic regions
        (c) use the top_k (=highest mean) of the spectrum feature vectors for IGE and gen
    """
    # load genes and intergenic examples
    (foo1, foo2, phi1_list, phi2_list, phi3_list, foo3) = load_genes(num_exms, signal, label, exm_id_intervals, min_lens=600, max_lens=1200)
    phi = co.matrix(phi1_list).trans()
    mgenes = np.sum(phi,1)/float(num_exms)
    mgenes /= max(abs(mgenes))

    (foo1, foo2, phi1_list, phi2_list, phi3_list, foo3) = load_intergenics(num_exms, signal, label, ige_intervals, min_lens=600, max_lens=1200)
    phi = co.matrix(phi1_list).trans()
    mige = np.sum(phi,1)/float(num_exms)
    mige /= max(abs(mige))

    # sort indices: most frequent or the last ones
    gen_inds = np.argsort(mgenes)
    ige_inds = np.argsort(mige)
    print('Calculated filter indices from {0} with {1} examples.'.format(fname,num_exms))
    print gen_inds
    print ige_inds
    inds = gen_inds[64-top_k:]
    inds = np.append(inds, ige_inds[64-top_k:])
    phi_inds = np.unique(inds)
    print phi_inds.shape
    print phi_inds
    #inds = {}
    #inds['ige'] = ige_inds
    #inds['gen'] = gen_inds
    #io.savemat('fergusonii_discr_feats.mat',inds)
    return (gen_inds[64-top_k:], ige_inds[64-top_k:], phi_inds)


def normalize_features(phi, ord=1):
    phi_norm = co.matrix(phi)
    for i in range(phi.size[1]):
        phi_norm[:,i] /= np.linalg.norm(phi_norm[:,i], ord=ord)
    return phi_norm


def perf_ocsvm(phi, marker, train, test, anom_prob, ord=1):
    #phi = phi[phi_inds.tolist(),:]
    print('(a) normalize features...')
    phi = normalize_features(phi, ord=ord)
    print('(a) Build kernel...')
    kern = Kernel.get_kernel(phi, phi)
    print('(b) Train OCSVM...')
    ocsvm = OCSVM(kern[train,train], C=1.0/(float(len(train))*(1.0-anom_prob)))
    ocsvm.train_dual()
    print('(c) Apply OCSVM...')
    (oc_as, foo) = ocsvm.apply_dual(kern[test, train[ocsvm.get_support_dual()]])
    (fpr, tpr, thres) = metric.roc_curve(co.matrix(marker)[test], oc_as)
    auc = metric.auc(fpr, tpr)
    print('(d) Return AUC={0}...'.format(auc))
    return auc


def perf_sad(test_inds, marker, train, test, anom_prob):
    # SAD annotation
    print('(a) Setup SAD...')
    lsvm = StructuredOCSVM(train, C=1.0/(train.samples*(1.0-anom_prob)))
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
    #for i in range(0,30):
    #    LENS = len(test.y[i])
    #    plt.plot(range(LENS),lats[i].trans() +(i-10)*10,'-r')
    #    plt.plot(range(LENS),test.y[i].trans() + 2 +(i-10)*10,'-b')
    #plt.show()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--anomalies", help="number of anomalies (default 10)", default=10, type =int)
    parser.add_argument("-n","--nominals", help="number of nominal data (default 190)", default=190, type =int)
    parser.add_argument("-r","--reps", help="number of repetitions (default 1)", default=1, type =int)
    parser.add_argument("-o","--output", help="output file name (default icml_pgm_b.mat)", default="icml_pgm_b.mat", type=str)
    parser.add_argument("-b","--basedir", help="base dir name (default nicococo/Code)", default="nicococo/Code", type=str)
    arguments = parser.parse_args(sys.argv[1:])

    print arguments.basedir
    #data = io.loadmat('/home/nico/Data/data.mat')
    data = io.loadmat('/home/{0}/ecoli/data.mat'.format(arguments.basedir))
    #data = io.loadmat('/home/nicococo/Code/anthracis/data.mat')
    #data = io.loadmat('/home/nicococo/Code/fergusonii/data.mat')
    exm_id_intervals = data['exm_id_intervals']
    exm_id = data['exm_id']
    label = data['label']
    signal = data['signal']

    # output file
    out_fname = arguments.output
    print arguments.output

    # find intergenic regions
    #ige_intervals = find_intergenic_regions(label, min_gene_dist=50)
    #intervals = {}
    #intervals['ige'] = ige_intervals
    #io.savemat('../ecoli/ige_intervals_2.mat',intervals)
    
    intervals = io.loadmat('/home/{0}/ecoli/ige_intervals_50.mat'.format(arguments.basedir))
    ige_intervals = intervals['ige']

    IGE_REGIONS = len(ige_intervals)
    
    EXMS = max(exm_id_intervals[:,0])
    DIMS = 4**3
    print('There are {0} gene examples.'.format(EXMS))

    NUM_TRAIN_GEN = arguments.anomalies
    NUM_TRAIN_IGE = arguments.nominals
    print("Anomalies={0} and Nominals={1}".format(NUM_TRAIN_GEN,NUM_TRAIN_IGE))
    
    NUM_TEST_GEN = 50
    NUM_TEST_IGE = 350

    NUM_COMB_GEN = NUM_TRAIN_GEN+NUM_TEST_GEN
    NUM_COMB_IGE = NUM_TRAIN_IGE+NUM_TEST_IGE
    anom_prob = float(NUM_TRAIN_GEN)/float(NUM_TRAIN_GEN+NUM_TRAIN_IGE)

    REPS = arguments.reps
    print("Repetitions = {0}".format(REPS))

    showPlots = True

    #(gen_inds, ige_inds, phi_inds) = feature_selection('/home/nicococo/Code/fergusonii/data.mat', top_k=24)
    (gen_inds, ige_inds, phi_inds) = feature_selection('/home/{0}/fergusonii/data.mat'.format(arguments.basedir), top_k=8)

    all_auc = {}
    all_res = {}
    for r in xrange(REPS):
        # shuffle genes and intergenics
        exm_id_intervals = np.random.permutation(exm_id_intervals)
        ige_intervals = np.random.permutation(ige_intervals)

        # load genes and intergenic examples
        (combX, combY, phi1_list, phi2_list, phi3_list, marker) = load_genes(NUM_COMB_GEN, signal, label, exm_id_intervals, min_lens=500, max_lens=1200)
        if len(combY)<NUM_COMB_GEN:
            raise Exception('Not enough genes ({0})! '.format(len(combY)))
        (X, Y, phis1, phis2, phis3, lbls) = load_intergenics(NUM_COMB_IGE, signal, label, ige_intervals, min_lens=500, max_lens=1200)
        if len(Y)<NUM_COMB_IGE:
            raise Exception('Not enough ige ({0})! '.format(len(Y)))
        combX.extend(X)
        combY.extend(Y)
        phi1_list.extend(phis1)
        phi2_list.extend(phis2)
        phi3_list.extend(phis3)
        marker.extend(lbls)
        EXMS = len(combY)
        combX = remove_mean(combX, DIMS)

        total_len = 0
        for i in range(EXMS):
            total_len += len(combY[i])
        print('---> Total length = {0}.'.format(total_len))

        trainX = combX[0:NUM_TRAIN_GEN]
        trainX.extend(X[0:NUM_TRAIN_IGE])
        trainY = combY[0:NUM_TRAIN_GEN]
        trainY.extend(Y[0:NUM_TRAIN_IGE])

        testX = combX[NUM_TRAIN_GEN:NUM_COMB_GEN]
        testX.extend(X[NUM_TRAIN_IGE:NUM_COMB_IGE])
        testY = combY[NUM_TRAIN_GEN:NUM_COMB_GEN]
        testY.extend(Y[NUM_TRAIN_IGE:NUM_COMB_IGE])

        state_map = []
        state_map.append(ige_inds.tolist())
        state_map.append([0,1,2,3])
        state_map.append([63,62,61,60,59])
        state_map.append(gen_inds.tolist())
        state_map.append(gen_inds.tolist())
        state_map.append(gen_inds.tolist())
        train = SOPGM(trainX, trainY, state_dims_map=state_map)
        test = SOPGM(testX, testY, state_dims_map=state_map)
        comb = SOPGM(combX, combY, state_dims_map=state_map)

        state_map = []
        state_map.append(range(64))
        state_map.append([0,1,2,3])
        state_map.append([63,62,61,60,59])
        state_map.append(range(64))
        state_map.append(range(64))
        state_map.append(range(64))
        train_full = SOPGM(trainX, trainY, state_dims_map=state_map)
        test_full = SOPGM(testX, testY, state_dims_map=state_map)

        inds_train = co.matrix(range(NUM_TRAIN_GEN) + range(NUM_COMB_GEN, NUM_COMB_GEN+NUM_TRAIN_IGE))
        inds_test = co.matrix(range(NUM_TRAIN_GEN,NUM_COMB_GEN) + range(NUM_COMB_GEN+NUM_TRAIN_IGE, NUM_COMB_GEN+NUM_COMB_IGE))

        # init result cache
        if not all_auc.has_key('SSVM (Full)'):
            # collect aucs
            all_auc['OcSvm Spectrum (1)'] = []
            all_auc['OcSvm Spectrum (2)'] = []
            all_auc['OcSvm Spectrum (3)'] = []
            all_auc['OcSvm Spectrum (FS)'] = []
            all_auc['SSVM (Full)'] = []
            all_auc['SSVM (FS)'] = []
            all_auc['HMAD (Full)'] = []
            all_auc['HMAD (FS)'] = []
            # collect fscores,..
            all_res['SSVM (Full)'] = []
            all_res['SSVM (FS)'] = []
            all_res['HMAD (Full)'] = []
            all_res['HMAD (FS)'] = []

        # structured output svm
        (auc, res) = perf_ssvm(inds_test, marker, train, test)
        all_auc['SSVM (FS)'].append(auc)
        all_res['SSVM (FS)'].append(res)
        (auc, res) = perf_ssvm(inds_test, marker, train_full, test_full)
        all_auc['SSVM (Full)'].append(auc)
        all_res['SSVM (Full)'].append(res)

        # spectrum kernel oc-svms
        auc = perf_ocsvm(co.matrix(phi1_list).trans(), marker, inds_train, inds_test, anom_prob)
        all_auc['OcSvm Spectrum (1)'].append(auc)
        auc = perf_ocsvm(co.matrix(phi2_list).trans(), marker, inds_train, inds_test, anom_prob)
        all_auc['OcSvm Spectrum (2)'].append(auc)
        auc = perf_ocsvm(co.matrix(phi3_list).trans(), marker, inds_train, inds_test, anom_prob)
        all_auc['OcSvm Spectrum (3)'].append(auc)

        # train one-class svm (use only filtered features)
        phi_fs = co.matrix(phi1_list).trans()
        phi_fs = phi_fs[phi_inds.tolist(),:]

        auc = perf_ocsvm(phi_fs, marker, inds_train, inds_test, anom_prob)
        all_auc['OcSvm Spectrum (FS)'].append(auc)

        (auc, res) = perf_sad(inds_test, marker, train, test, anom_prob)
        all_auc['HMAD (FS)'].append(auc)
        all_res['HMAD (FS)'].append(res)
        (auc, res) = perf_sad(inds_test, marker, train_full, test_full, anom_prob)
        all_auc['HMAD (Full)'].append(auc)
        all_res['HMAD (Full)'].append(res)


    print '##############################################'
    print  out_fname
    print '##############################################'
    print NUM_COMB_GEN
    print NUM_COMB_IGE
    print '##############################################'
    print total_len 
    print anom_prob
    print '##############################################'
    print ige_inds
    print gen_inds  
    print '##############################################'
    print all_auc
    print '##############################################'
    print all_res
    print '##############################################'
    # store result as a file
    data = {}
    data['auc'] = all_auc
    data['res'] = all_res

    io.savemat(out_fname, data)
    print('finished')
