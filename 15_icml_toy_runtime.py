import time as timer
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
    X = normalize_sequence_data(X)
    return (SOHMM(X[0:num_train],Y[0:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label)


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


def build_histograms(data, phi, num_train, bins=2, ord=2):
    # first num_train phis are used for estimating
    # histogram boundaries.
    N = len(data)
    (F, LEN) = data[0].size

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
    kern = Kernel.get_kernel(phi_hist, phi_hist)
    return kern, phi_hist


def build_seq_kernel(data, ord=2, type='linear', param=1.0):
    # all sequences have the same length
    N = len(data)
    (F, LEN) = data[0].size
    phi = co.matrix(0.0, (F*LEN, N))
    for i in xrange(N):
        for f in xrange(F):
            phi[(f*LEN):(f*LEN)+LEN,i] = data[i][f,:].trans()
        if ord>=1:
            phi[:,i] /= np.linalg.norm(phi[:,i], ord=ord)
    kern = Kernel.get_kernel(phi, phi, type=type, param=param)
    return kern, phi

def build_kernel(data, num_train, bins=2, ord=2, typ='linear', param=1.0):
    if typ=='hist':
        foo, phi = build_seq_kernel(data, ord=-1)
        return build_histograms(data, phi, num_train, bins=param, ord=ord)
    elif typ=='':
        return -1,-1
    else:
        return build_seq_kernel(data, ord=ord, type=typ.lower(), param=param)

def test_bayes(phi, kern, train, test, num_train, anom_prob, labels):
    startTime = timer.time()
    # bayes classifier
    (DIMS, N) = phi.size
    w_bayes = co.matrix(-1.0, (DIMS, 1))
    #pred = w_bayes.trans()*phi[:,num_train:]
    #(fpr, tpr, thres) = metric.roc_curve(labels[num_train:], pred.trans())
    return timer.time() - startTime

def test_ocsvm(phi, kern, train, test, num_train, anom_prob, labels):
    startTime = timer.time()
    ocsvm = OCSVM(kern[:num_train,:num_train], C=1.0/(num_train*anom_prob))
    msg = ocsvm.train_dual()
    return timer.time() - startTime

def test_hmad(phi, kern, train, test, num_train, anom_prob, labels, zero_shot=False):
    startTime = timer.time()
    # train structured anomaly detection
    sad = StructuredOCSVM(train, C=1.0/(num_train*anom_prob))
    (lsol, lats, thres) = sad.train_dc(max_iter=60, zero_shot=zero_shot)
    return timer.time() - startTime



if __name__ == '__main__':
    LENS = 600
    EXMS = 1100
    EXMS_TRAIN = 400
    ANOM_PROB = 0.1
    REPS = 50
    BLOCK_LEN = 120

    BLOCKS = [100,200,400,600,800,1000]
    #BLOCKS = [400]

    methods = ['Bayes' ,'HMAD','OcSvm','OcSvm','OcSvm','OcSvm','OcSvm','OcSvm','OcSvm','OcSvm']
    kernels = ['Linear',''    ,'RBF'  ,'RBF'  ,'RBF'  ,'Hist' ,'Hist' ,'Hist' ,'Linear','Linear']
    kparams = [''      ,''    ,  0.1  ,  1.0  ,  10.0 , 4     , 8     , 10    , ''     , '']
    ords    = [+1      , 1    ,  1    ,  1    ,  1    , 1     , 1     , 1     , 1      , 2]

    #methods = ['OcSvm','OcSvm','OcSvm']
    #kernels = ['RBF'  ,'RBF'  ,'RBF'  ]
    #kparams = [  10.1 ,  1000.0  ,  0.1  ]
    #ords    = [  1    ,  1    ,  1    ]

    #methods = ['Bayes','Bayes']
    #kernels = ['Linear'  ,'Linear'  ]
    #kparams = [  1 ,  1]
    #ords    = [  1 ,  2]

    # collected means
    res = []
    for r in xrange(REPS):
        for b in xrange(len(BLOCKS)):
            (train, test, comb, labels) = get_model(BLOCKS[b]+1, BLOCKS[b], LENS, BLOCK_LEN, blocks=1, anomaly_prob=ANOM_PROB)
            for m in range(len(methods)):
                name = 'test_{0}'.format(methods[m].lower())
                (kern, phi) = build_kernel(comb.X, BLOCKS[b], ord=ords[m], typ=kernels[m].lower(), param=kparams[m])
                print('Calling {0}'.format(name))
                time = eval(name)(phi, kern, train, test, BLOCKS[b], ANOM_PROB, labels)
                print('-------------------------------------------------------------------------------')
                print 
                print('Iter {0}/{1} in block {2}/{3} for method {4} ({5}/{6}) got TIME = {7}.'.format(r+1,REPS,b+1,len(BLOCKS),name,m+1,len(methods),time))
                print
                print('-------------------------------------------------------------------------------')

                if len(res)<=b:
                    res.append([])
                mlist = res[b]
                if len(mlist)<=m:
                    mlist.append([])
                cur = mlist[m]
                cur.append(time)

    print('RESULTS >-----------------------------------------')
    print 
    times = np.ones((len(methods),len(BLOCKS)))
    stds = np.ones((len(methods),len(BLOCKS)))
    varis = np.ones((len(methods),len(BLOCKS)))
    names = []

    for b in range(len(BLOCKS)):
        print("BLOCKS={0}:".format(BLOCKS[b]))
        for m in range(len(methods)):
            time = np.mean(res[b][m])
            std = np.std(res[b][m])
            var = np.var(res[b][m])
            times[m,b] = time
            stds[m,b] = std
            varis[m,b] = var
            kname = '' 
            if kernels[m]=='RBF' or kernels[m]=='Hist':
                kname = ' ({0} {1})'.format(kernels[m],kparams[m])
            elif kernels[m]=='Linear':
                kname = ' ({0})'.format(kernels[m])
            name = '{0}{1} [{2}]'.format(methods[m],kname,ords[m])
            if len(names)<=m:
                names.append(name)
            print("   m={0}: Time={1} STD={2} VAR={3}".format(name,time,std,var))
        print


    print times
    # store result as a file
    data = {}
    data['LENS'] = LENS
    data['EXMS'] = EXMS
    data['EXMS_TRAIN'] = EXMS_TRAIN
    data['ANOM_PROB'] = ANOM_PROB
    data['REPS'] = REPS
    data['BLOCKS'] = BLOCKS

    data['methods'] = methods
    data['kernels'] = kernels
    data['kparams'] = kparams
    data['ords'] = ords

    data['res'] = res
    data['times'] = times
    data['stds'] = stds
    data['varis'] = varis
    data['names'] = names

    io.savemat('15_icml_toy_runtime_b0.mat',data)

    print('finished')
