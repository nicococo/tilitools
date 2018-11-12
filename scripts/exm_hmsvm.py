import numpy as np
import sklearn.metrics as metric

from profiler import print_profiles
from utils_kernel import get_kernel
from utils_data import get_2state_gaussian_seq, get_2state_anom_seq

from ocsvm_dual_qp import OcSvmDualQP
from latent_ocsvm import LatentOCSVM

from so_hmm import SOHMM


def get_model(num_exm, num_train, lens, feats, anomaly_prob=0.15):
    print('Generating {0} sequences, {1} for training, each with {2} anomaly probability.'.format(num_exm, num_train, anomaly_prob))
    mean = np.zeros(feats)
    cnt = 0
    X = []
    Y = []
    label = []
    for i in range(num_exm):
        exm, lbl, marker = get_2state_gaussian_seq(lens, dims=feats, anom_prob=anomaly_prob)
        mean += np.ones((1, lens)).dot(exm.T).reshape(feats) / np.float(lens)
        X.append(exm)
        Y.append(lbl)
        label.append(marker)
    mean = mean / np.float(num_exm)
    for i in range(num_exm):
        X[i] = X[i] - mean.reshape((feats, 1)).repeat(lens, axis=1)
    return SOHMM(X[:num_train],Y[:num_train]), SOHMM(X[num_train:],Y[num_train:]), SOHMM(X,Y), label


def calc_feature_vecs(data):
    # ASSUME that all sequences have the same length!
    N = len(data)
    F, LEN = data[0].shape
    phi = np.zeros((F*LEN, N))
    for i in range(N):
        for f in range(F):
            phi[(f*LEN):(f*LEN)+LEN, i] = data[i][f,:].T
    return phi


def experiment_anomaly_detection(train, test, comb, num_train, anom_prob, labels):
    # train one-class svm
    phi = calc_feature_vecs(comb.X)
    kern = get_kernel(phi[:,0:num_train], phi[:,0:num_train])
    ocsvm = OcSvmDualQP(kern, anom_prob)
    ocsvm.fit()
    kern = get_kernel(phi, phi)
    oc_as = ocsvm.apply(kern[num_train:,ocsvm.get_support_dual()])
    fpr, tpr, thres = metric.roc_curve(labels[num_train:], oc_as)
    base_auc = metric.auc(fpr, tpr)

    # train structured anomaly detection
    sad = LatentOCSVM(train, anom_prob)
    sad.fit(max_iter=40)
    pred_vals, pred_lats = sad.apply(test)
    fpr, tpr, thres = metric.roc_curve(labels[num_train:], pred_vals)
    auc = metric.auc(fpr, tpr)
    return auc, base_auc


if __name__ == '__main__':
    DIMS = 2
    LENS = 250
    EXMS = 600
    EXMS_TRAIN = 100
    ANOM_PROB = 0.01
    REPS = 5

    # anomaly detection experiment
    aucs = np.zeros((2, REPS))
    for r in range(REPS):
        train, test, comb, labels = get_model(EXMS, EXMS_TRAIN, LENS, DIMS, ANOM_PROB)
        aucs[0, r], aucs[1, r] = experiment_anomaly_detection(train, test, comb, EXMS_TRAIN, ANOM_PROB, labels)

    print(aucs)
    print('####################')
    print('  SAD={0:1.2f} +/- {1:1.2f}'.format(np.mean(aucs[0, :]), np.var(aucs[0, :])))
    print('OCSVM={0:1.2f} +/- {1:1.2f}'.format(np.mean(aucs[1, :]), np.var(aucs[1, :])))
    print('####################')

    print_profiles()
    print('finished')