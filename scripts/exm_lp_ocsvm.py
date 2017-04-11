import numpy as np
import matplotlib.pyplot as plt

from tilitools.lp_ocsvm_primal_sgd import LpOcSvmPrimalSGD
from tilitools.ocsvm_dual_qp import OcSvmDualQP
from tilitools.utils_kernel import get_kernel, center_kernel, normalize_kernel


if __name__ == '__main__':
    # kernel parameter and type
    kparam = 4.
    ktype = 'linear'

    # generate raw training data
    Dtrain = np.random.randn(3, 1000)*0.5
    Dtrain[2, :] = 1.0
    for i in range(Dtrain.shape[1]):
        Dtrain[:2, i] /= np.linalg.norm(Dtrain[:2, i])
    kernel = get_kernel(Dtrain, Dtrain, ktype, kparam)

    svm = OcSvmDualQP(kernel, 0.5)
    svm.fit()

    skl_svm = LpOcSvmPrimalSGD(pnorm=2., nu=.5)
    skl_svm.fit(Dtrain)

    delta = 0.1
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    (sx,sy) = X.shape
    Xf = np.reshape(X,(1,sx*sy))
    Yf = np.reshape(Y,(1,sx*sy))
    Dtest = np.ones((3, sx*sy))
    Dtest[:2, :] = np.append(Xf, Yf, axis=0)

    # build test kernel
    kernel = get_kernel(Dtest, Dtrain[:,svm.get_support_dual()], ktype, kparam)
    res = svm.apply(kernel)
    skl_res = skl_svm.apply(Dtest)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    Z = np.reshape(res,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0, svm.get_support_dual()], Dtrain[1, svm.get_support_dual()], 40, c='k')
    plt.scatter(Dtrain[0, svm.get_outliers()], Dtrain[1, svm.get_outliers()], 40, c='c')
    plt.scatter(Dtrain[0,:], Dtrain[1,:], 10)

    plt.subplot(1, 2, 2)
    Z = np.reshape(skl_res,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0,:], Dtrain[1,:], 10)
    plt.show()

    print('finished')
