import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM

from tilitools.ocsvm_dual_qp import OcSvmDualQP
from tilitools.utils_kernel import get_kernel, center_kernel, normalize_kernel


if __name__ == '__main__':
    # kernel parameter and type
    kparam = 4.
    ktype = 'rbf'

    # generate raw training data
    Dtrain = np.random.randn(2, 100)
    # build kernel
    kernel = get_kernel(Dtrain, Dtrain, ktype, kparam)
    # kernel = center_kernel(kernel)
    kernel = normalize_kernel(kernel)
    # train svdd

    svm = OcSvmDualQP(kernel, 0.1)
    svm.fit()

    skl_svm = OneClassSVM(kernel='precomputed', nu=0.1, shrinking=False, verbose=True, tol=1e-3)
    skl_svm.fit(kernel)
    dists = skl_svm.decision_function(kernel)
    print(skl_svm._intercept_)
    print(skl_svm.support_.shape)
    print(np.sum(skl_svm.dual_coef_))
    k = kernel[skl_svm.support_, :]
    k = k[:, skl_svm.support_]
    print(0.5*skl_svm.dual_coef_.dot(k).dot(skl_svm.dual_coef_.T))
    print(skl_svm.dual_coef_)

    print(np.sum(dists < 0.))
    print(np.sum(dists <= 0.))

    skl_outliers = np.where(dists < 0.)[0]
    print(skl_outliers)
    print(svm.outliers)

    delta = 0.1
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    sx, sy = X.shape
    Xf = np.reshape(X, (1, sx*sy))
    Yf = np.reshape(Y, (1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)
    foo = 3. * delta

    # build test kernel
    kernel = get_kernel(Dtest, Dtrain[:,svm.get_support_dual()], ktype, kparam)

    res = svm.apply(kernel)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    Z = np.reshape(res,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.])
    plt.scatter(Dtrain[0, svm.get_support_dual()], Dtrain[1, svm.get_support_dual()], 40, c='k')
    plt.scatter(Dtrain[0, svm.get_outliers()], Dtrain[1, svm.get_outliers()], 40, c='c')
    plt.scatter(Dtrain[0,:], Dtrain[1,:], 10)

    plt.subplot(1, 2, 2)
    Z = np.reshape(res,(sx,sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.])
    plt.scatter(Dtrain[0, skl_outliers], Dtrain[1, skl_outliers], 40, c='c')
    plt.scatter(Dtrain[0,:], Dtrain[1,:], 10)
    plt.show()

    print('finished')
