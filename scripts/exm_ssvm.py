import cvxopt as co
import numpy as np

from tilitools.ssvm import SSVM
from tilitools.so_multiclass import SOMultiClass
from tilitools import utils_data
from tilitools import utils


if __name__ == '__main__':
    # generate raw training data
    Dtrain1 = utils_data.get_gaussian(1000, dims=2, means=[4.0, 2.0], vars=[1.0, 0.3])
    Dtrain2 = utils_data.get_gaussian(100, dims=2, means=[-2.0, 1.0], vars=[0.3, 1.3])
    Dtrain3 = utils_data.get_gaussian(100, dims=2, means=[3.0, -1.0], vars=[0.3, 0.3])
    Dtrain4 = utils_data.get_gaussian(50, dims=2, means=[6.0, -3.0], vars=[0.2, 0.1])

    Dtrain = np.concatenate((Dtrain1.T, Dtrain2.T, Dtrain3.T, Dtrain4.T)).T
    Dtrain = np.concatenate((Dtrain, np.ones((1250, 1)).T))
    print Dtrain.shape
    Dy = np.zeros(Dtrain.shape[1], dtype=np.int)
    Dy[1000:1100] = 1
    Dy[1100:1200] = 2
    Dy[1200:] = 3

    # generate structured object
    sobj = SOMultiClass(Dtrain, np.unique(Dy).size, Dy)

    # train structured output support vector machine
    ssvm = SSVM(sobj, 1.0)
    ws, slacks = ssvm.train()

    # generate test data grid
    delta = 0.1
    x = np.arange(-4.0, 8.0, delta)
    y = np.arange(-4.0, 8.0, delta)
    X, Y = np.meshgrid(x, y)
    sx, sy = X.shape
    Xf = np.reshape(X, (1, sx*sy))
    Yf = np.reshape(Y, (1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)
    Dtest = np.append(Dtest, np.reshape([1.0]*(sx*sy), (1,sx*sy)), axis=0)

    # generate structured object
    predsobj = SOMultiClass(Dtest, np.unique(Dy).size)
    res, cls = ssvm.apply(predsobj)

    utils.print_profiles()

    print('finished')