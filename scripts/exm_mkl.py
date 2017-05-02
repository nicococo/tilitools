import cvxopt as co
import numpy as np
import matplotlib.pyplot as plt

from tilitools.utils import print_profiles
from tilitools.ssad_convex import ConvexSSAD
from tilitools.lp_mkl_wrapper import MKLWrapper
from tilitools.utils_kernel import get_kernel, center_kernel, normalize_kernel


if __name__ == '__main__':
    """
    MKL is a wrapper around a specific method (here SSAD) and
    takes a list of kernels as input argument.
    support vectors (N_1 <= N).
    """
    P_NORM = 1.1 # mixing coefficient lp-norm regularizer
    N_pos = 100
    N_neg = 100
    N_unl = 10

    # 1. STEP: GENERATE DATA
    # 1.1. generate training labels
    yp = co.matrix( 1, (1, N_pos),'i')
    yu = co.matrix( 0, (1, N_unl),'i')
    yn = co.matrix(-1, (1, N_neg),'i')
    Dy = co.matrix([[yp], [yu], [yn], [yn], [yn], [yn]])
    Dy = np.array(Dy)
    Dy = Dy.reshape((Dy.size))

    # 1.2. generate training data
    co.setseed(11)
    Dtrainp = co.normal(2, N_pos)*0.4
    Dtrainu = co.normal(2, N_unl)*0.4
    Dtrainn = co.normal(2, N_neg)*0.1
    Dtrain21 = Dtrainn-1
    Dtrain21[0, :] = Dtrainn[0, :] + 1
    Dtrain22 = -Dtrain21

    # 1.3. concatenate training data
    Dtrain = co.matrix([[Dtrainp], [Dtrainu], [Dtrainn+1.0], [Dtrainn-1.0], [Dtrain21], [Dtrain22]])
    Dtrain = np.array(Dtrain)

    # 1.4. generate test data on a grid
    delta = 0.25
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    sx, sy = X.shape
    Xf = np.reshape(X, (1, sx*sy))
    Yf = np.reshape(Y, (1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)

    # 1.4. build the training kernels:
    # - same training sample for each kernel = they have the same size
    # - each kernel captures a feature representation of the samples Dtrain
    #   e.g., kernel1 is a BOW kernel, kernel2 a Lexical Diversity kernel
    #   here: kernel1 und kernel2 are Gaussian kernels with different shape parameters
    # 	and kernel3 is a simple linear kernel
    train_kernels = []
    test_kernels = []
    rbf_vals = [0.01, 1., 10., 100.]
    for val in rbf_vals:
        data = np.concatenate((Dtrain, Dtest), axis=1)
        print data.shape
        kernel = get_kernel(data, data, type='rbf', param=val)
        kernel = center_kernel(kernel)
        kernel = normalize_kernel(kernel)

        train_kernel = kernel[:Dtrain.shape[1], :].copy()
        test_kernel = kernel[:Dtrain.shape[1], :].copy()
        train_kernels.append(train_kernel[:, :Dtrain.shape[1]])
        test_kernels.append(test_kernel[:, Dtrain.shape[1]:].T)

    # MKL: (default) use SSAD
    ad = ConvexSSAD([], Dy, 1.0, 1.0, 1.0 / (100 * 0.05), 1.0)

    # 2. STEP: TRAIN WITH A LIST OF KERNELS
    ssad = MKLWrapper(ad, train_kernels, P_NORM)
    ssad.fit()

    # 3. TEST
    thres = ssad.get_threshold()
    pred = ssad.apply(train_kernels)
    pred = pred[ssad.get_support_dual()]
    res = ssad.apply(test_kernels)

    # 4. STEP: PLOT RESULTS

    # 4.1. plot training and test data
    Z = np.reshape(res, (sx, sy))
    plt.figure()
    plt.contourf(X, Y, Z, 20)
    plt.contour(X, Y, Z, [ssad.get_threshold()])
    plt.scatter(Dtrain[0, ssad.get_support_dual()], Dtrain[1, ssad.get_support_dual()], 60, c='w')
    plt.scatter(Dtrain[0,N_pos:N_pos+N_unl-1], Dtrain[1,N_pos:N_pos+N_unl-1], 10, c='g')
    plt.scatter(Dtrain[0,0:N_pos], Dtrain[1,0:N_pos], 20, c='r')
    plt.scatter(Dtrain[0,N_pos+N_unl:], Dtrain[1,N_pos+N_unl:], 20, c='b')

    # 4.2. plot the influence of each kernel
    plt.figure()
    plt.bar([i+1 for i in range(4)], ssad.get_mixing_coefficients())
    plt.show()

    print_profiles()
    print('finished')