import numpy as np
import matplotlib.pyplot as plt

from ssvm import SSVM
from latent_svdd import LatentSVDD
from latent_ocsvm import LatentOCSVM
from latent_pca import LatentPCA

from so_multiclass import SOMultiClass


if __name__ == '__main__':
    NUM_CLASSES = 3
    NUM_DATA = 40  # per class

    # generate raw training data
    Dy = np.zeros(NUM_CLASSES*NUM_DATA, dtype=np.int)
    Dtrain = np.ones((NUM_CLASSES*NUM_DATA, 3))
    for i in range(NUM_CLASSES):
        mean = (np.random.rand(2)-0.5)*12.
        Dtrain[i*NUM_DATA:(i+1)*NUM_DATA, :2] = np.random.multivariate_normal(mean, 1.*np.random.rand()*np.eye(2), size=NUM_DATA)
        Dy[i*NUM_DATA:(i+1)*NUM_DATA] = i
    # generate structured object
    sobj = SOMultiClass(Dtrain.T, y=Dy , classes=NUM_CLASSES)

    # unsupervised methods
    lsvdd = LatentSVDD(sobj, 0.9)
    lsvdd.fit()
    spca = LatentPCA(sobj)
    spca.fit()

    socsvm = LatentOCSVM(sobj, .2)
    socsvm.fit()

    # supervised methods
    ssvm = SSVM(sobj)
    ssvm.train()

    # generate test data grid
    delta = 0.2
    x = np.arange(-8.0, 8.0, delta)
    y = np.arange(-8.0, 8.0, delta)
    X, Y = np.meshgrid(x, y)
    (sx,sy) = X.shape
    Xf = np.reshape(X,(1,sx*sy))
    Yf = np.reshape(Y,(1,sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)
    Dtest = np.append(Dtest, np.ones((1, sx*sy)), axis=0)
    print(Dtest.shape)

    # generate structured object
    predsobj = SOMultiClass(Dtest, NUM_CLASSES)

    # for all methods
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2,4,i+1)

        if i==0:
            plt.title("LatentSVDD")
            scores, lats = lsvdd.apply(predsobj)
        if i==1:
            plt.title("StructPCA")
            (scores,lats) = spca.apply(predsobj)
        if i==2:
            plt.title("StructOCSVM")
            scores,lats = socsvm.apply(predsobj)
        if i==3:
            plt.title("SSVM")
            scores, lats = ssvm.apply(predsobj)

        # plot scores
        Z = np.reshape(scores, (sx,sy))
        plt.contourf(X, Y, Z)
        plt.grid()
        plt.scatter(Dtrain[:, 0], Dtrain[:, 1], 10)

        # plot latent variable
        Z = np.reshape(lats, (sx,sy))
        plt.subplot(2, 4, i+4+1)
        plt.contourf(X, Y, Z)
        plt.grid()
        plt.scatter(Dtrain[:, 0], Dtrain[:, 1], 10)
    plt.show()

    print('finished')