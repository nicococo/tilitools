import numpy as np
import matplotlib.pyplot as plt

from tilitools.lp_ocsvm_primal_sgd import LpOcSvmPrimalSGD
from tilitools.ocsvm_dual_qp import OcSvmDualQP
from tilitools.utils_kernel import get_kernel, center_kernel, normalize_kernel
from tilitools.utils import print_profiles


if __name__ == '__main__':
    # kernel parameter and type
    ktype = 'linear'
    nu = 0.25

    # generate raw training data
    Dtrain = (np.random.rand(2, 100)-0.)*3.
    Dtrain[1, :] = (np.random.rand(1, Dtrain.shape[1])-0.)*0.9
    # for i in range(Dtrain.shape[1]):
    #     Dtrain[:2, i] /= np.linalg.norm(Dtrain[:2, i])
    kernel = get_kernel(Dtrain, Dtrain, ktype)

    svm = OcSvmDualQP(kernel, nu)
    svm.fit()

    skl_svm = LpOcSvmPrimalSGD(pnorm=2., nu=nu)
    skl_svm.fit(Dtrain)
    skl_svm.fit(Dtrain)

    delta = 0.1
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    (sx,sy) = X.shape
    Xf = np.reshape(X,(1,sx*sy))
    Yf = np.reshape(Y,(1,sx*sy))
    Dtest = np.ones((2, sx*sy))
    Dtest[:2, :] = np.append(Xf, Yf, axis=0)
    # for i in range(Dtest.shape[1]):
    #     Dtest[:2, i] /= np.linalg.norm(Dtest[:2, i])

    # build test kernel
    kernel = get_kernel(Dtest, Dtrain[:,svm.get_support_dual()], ktype)
    res = svm.apply(kernel)
    skl_res = skl_svm.apply(Dtest)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    Z = np.reshape(res,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0, svm.get_support_dual()], Dtrain[1, svm.get_support_dual()], 40, c='k')
    plt.scatter(Dtrain[0, svm.get_outliers()], Dtrain[1, svm.get_outliers()], 40, c='c')
    plt.scatter(Dtrain[0,:], Dtrain[1,:], 10)

    plt.subplot(1, 2, 2)
    Z = np.reshape(skl_res,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0, skl_svm.get_outliers()], Dtrain[1, skl_svm.get_outliers()], 40, c='c')
    plt.scatter(Dtrain[0,:], Dtrain[1,:], 10)
    plt.show()

    print_profiles()
    print('finished')
