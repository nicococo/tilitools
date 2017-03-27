import cvxopt as co
import numpy as np
import matplotlib.pyplot as plt

from ssvm import SSVM
from so_multiclass import SOMultiClass
from toydata import ToyData


if __name__ == '__main__':
    # generate raw training data
    Dtrain1 = ToyData.get_gaussian(1000, dims=2, means=[4.0,2.0], vars=[1.0,0.3])
    Dtrain2 = ToyData.get_gaussian(100, dims=2, means=[-2.0,1.0], vars=[0.3,1.3])
    Dtrain3 = ToyData.get_gaussian(100, dims=2, means=[3.0,-1.0], vars=[0.3,0.3])
    Dtrain4 = ToyData.get_gaussian(50, dims=2, means=[6.0,-3.0], vars=[0.2,0.1])

    Dtrain = co.matrix([[Dtrain1], [Dtrain2], [Dtrain3], [Dtrain4]])
    Dtrain = co.matrix([[Dtrain.trans()],[co.matrix(1.0,(1250,1))]]).trans()
    Dy = co.matrix([[co.matrix(0,(1,1000))], [co.matrix(1,(1,100))], [co.matrix(2,(1,100))], [co.matrix(3,(1,50))]])

    # generate structured object
    sobj = SOMultiClass(Dtrain, 4, Dy)

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
    predsobj = SOMultiClass(co.matrix(Dtest), 4)
    res, cls = ssvm.apply(predsobj)

    # nice visualization
    Z = np.reshape(cls,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.scatter(Dtrain[0, :], Dtrain[1, :], 10)
    plt.show()

    print('finished')