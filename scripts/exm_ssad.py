import cvxopt as co
import numpy as np
import matplotlib.pyplot as plt

from tilitools.ssad_convex import ConvexSSAD
from tilitools.utils_kernel import get_kernel


if __name__ == '__main__':
    # example constants (training set size and splitting)
    k_type = 'rbf'
    # attention: this is the shape parameter of a Gaussian
    # which is 1/sigma^2
    k_param = 2.4

    N_pos = 10
    N_neg = 10
    N_unl = 10

    # generate training labels
    Dy = np.zeros(N_pos+N_neg+N_unl, dtype=np.int)
    Dy[:N_pos] = 1
    Dy[N_pos+N_unl:] = -1

    # generate training data
    co.setseed(11)
    Dtrainp = co.normal(2,N_pos)*0.6
    Dtrainu = co.normal(2,N_unl)*0.6
    Dtrainn = co.normal(2,N_neg)*0.6
    Dtrain21 = Dtrainn-1
    Dtrain21[0,:] = Dtrainn[0,:]+1
    Dtrain22 = -Dtrain21

    # training data
    Dtrain = co.matrix([[Dtrainp], [Dtrainu], [Dtrainn+0.8]])

    Dtrain = np.array(Dtrain)

    # build the training kernel
    kernel = get_kernel(Dtrain, Dtrain, type=k_type, param=k_param)

    # use SSAD
    ssad = ConvexSSAD(kernel, Dy, 1./(10.*0.1), 1./(10.*0.1), 1., 1/(10.*0.1))
    ssad.fit()

    # generate test data from a grid for nicer plots
    delta = 0.1
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    (sx,sy) = X.shape
    Xf = np.reshape(X,(1,sx*sy))
    Yf = np.reshape(Y,(1,sx*sy))
    Dtest = np.append(Xf,Yf,axis=0)
    print(Dtest.shape)

    # build the test kernel
    kernel = get_kernel(Dtest, Dtrain[:, ssad.svs], type=k_type, param=k_param)
    res = ssad.apply(kernel)

    # make a nice plot of it
    Z = np.reshape(res,(sx,sy))
    plt.contourf(X, Y, Z, 20, cmap='Blues')
    plt.colorbar()
    plt.contour(X, Y, Z, np.linspace(0.0, np.max(res), 10))
    # plt.contour(X, Y, Z, [-0.6, 0., 0.6])
    plt.scatter(Dtrain[0, ssad.svs], Dtrain[1, ssad.svs], 60, c='w')

    plt.scatter(Dtrain[0,N_pos:N_pos+N_unl-1],Dtrain[1,N_pos:N_pos+N_unl-1], 10, c='g')
    plt.scatter(Dtrain[0,0:N_pos],Dtrain[1,0:N_pos], 20, c='r')
    plt.scatter(Dtrain[0,N_pos+N_unl:],Dtrain[1,N_pos+N_unl:], 20, c='b')

    plt.show()
