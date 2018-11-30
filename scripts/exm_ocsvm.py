import numpy as np
import matplotlib.pyplot as plt

from tilitools.lp_ocsvm_primal_sgd import LpOcSvmPrimalSGD
from tilitools.ocsvm_dual_qp import OcSvmDualQP
from tilitools.huber_ocsvm_primal import HuberOcsvmPrimal
from tilitools.utils_kernel import get_kernel
from tilitools.profiler import print_profiles


if __name__ == '__main__':
    nu = 0.25

    # generate raw training data
    Dtrain = np.random.rand(2, 200)*3.
    Dtrain[1, :] = np.random.rand(1, Dtrain.shape[1])*0.9
    kernel = get_kernel(Dtrain, Dtrain, 'linear')

    huber = HuberOcsvmPrimal(nu)
    huber.fit(Dtrain)

    svm = OcSvmDualQP(kernel, nu)
    svm.fit()

    skl_svm = LpOcSvmPrimalSGD(pnorm=2., nu=nu)
    skl_svm.fit(Dtrain)
    skl_svm.fit(Dtrain)

    delta = 0.1
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    sx, sy = X.shape
    Xf = np.reshape(X, (1, sx*sy))
    Yf = np.reshape(Y, (1, sx*sy))
    Dtest = np.ones((2, sx*sy))
    Dtest[:2, :] = np.append(Xf, Yf, axis=0)

    kernel = get_kernel(Dtest, Dtrain[:, svm.get_support_dual()], 'linear')
    res = svm.apply(kernel)
    skl_res = skl_svm.apply(Dtest)
    huber_res = huber.apply(Dtest)

    plt.figure(1)
    plt.subplot(1, 3, 1)
    Z = np.reshape(res, (sx, sy))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0, svm.get_support_dual()], Dtrain[1, svm.get_support_dual()], 40, c='k')
    plt.scatter(Dtrain[0, svm.get_outliers()], Dtrain[1, svm.get_outliers()], 40, c='c')
    plt.scatter(Dtrain[0, :], Dtrain[1, :], 10)

    plt.subplot(1, 3, 2)
    Z = np.reshape(skl_res, (sx, sy))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0, skl_svm.get_outliers()], Dtrain[1, skl_svm.get_outliers()], 40, c='c')
    plt.scatter(Dtrain[0, :], Dtrain[1, :], 10)
    plt.show()

    plt.subplot(1, 3, 3)
    Z = np.reshape(huber_res, (sx, sy))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.contour(X, Y, Z, [0.], linewidth=4., color='k')
    plt.scatter(Dtrain[0,: ], Dtrain[1, :], 10)

    print_profiles()
    print('finished')
