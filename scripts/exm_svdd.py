import matplotlib.pyplot as plt
import numpy as np

from tilitools.svdd_dual_qp import SvddDualQP
from tilitools.svdd_primal_sgd import SvddPrimalSGD


if __name__ == '__main__':
    nu = 0.15  # outlier fraction

    # generate raw training data
    Dtrain = np.random.randn(2, 1000)
    Dtrain /= np.max(np.abs(Dtrain))

    # train dual svdd
    svdd = SvddDualQP('linear', 0.1, nu)
    svdd.fit(Dtrain)

    # train primal svdd
    psvdd = SvddPrimalSGD(nu)
    psvdd.fit(Dtrain, max_iter=1000, prec=1e-4)

    # print solutions
    print('\n  dual-svdd: obj={0}  T={1}.'.format(svdd.pobj, svdd.radius2))
    print('primal-svdd: obj={0}  T={1}.\n'.format(psvdd.pobj, psvdd.radius2))

    # generate test data grid
    delta = 0.1
    x = np.arange(-2.0-delta, 2.0+delta, delta)
    y = np.arange(-2.0-delta, 2.0+delta, delta)
    X, Y = np.meshgrid(x, y)
    (sx, sy) = X.shape
    Xf = np.reshape(X,(1, sx*sy))
    Yf = np.reshape(Y,(1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)
    if Dtrain.shape[0] > 2:
        Dtest = np.append(Dtest, np.random.randn(Dtrain.shape[0]-2, sx*sy), axis=0)
    print(Dtest.shape)

    res = svdd.predict(Dtest)
    pres = psvdd.predict(Dtest)

    # nice visualization
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title('Dual QP SVDD')
    Z = np.reshape(res,(sx, sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.0], linewidths=3.0, colors='k')
    plt.scatter(Dtrain[0, svdd.get_support_inds()], Dtrain[1, svdd.get_support_inds()], 40, c='k')
    plt.scatter(Dtrain[0, :], Dtrain[1, :],10)
    plt.xlim((-2., 2.))
    plt.ylim((-2., 2.))
    plt.yticks(range(-2, 2), [])
    plt.xticks(range(-2, 2), [])

    plt.subplot(1, 2, 2)
    plt.title('Primal Subgradient SVDD')
    Z = np.reshape(pres,(sx, sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.0], linewidths=3.0, colors='k')
    plt.scatter(Dtrain[0, :], Dtrain[1, :], 10)
    plt.xlim((-2., 2.))
    plt.ylim((-2., 2.))
    plt.yticks(range(-2, 2), [])
    plt.xticks(range(-2, 2), [])

    plt.show()

    print('finished')