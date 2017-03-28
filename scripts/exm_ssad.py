import cvxopt as co
import numpy as np
import matplotlib.pyplot as plt

from tilitools.ssad_convex import ConvexSSAD
from tilitools.kernel import get_diag_kernel, get_kernel


if __name__ == '__main__':
    # example constants (training set size and splitting)
    k_type = 'rbf'
    # attention: this is the shape parameter of a Gaussian
    # which is 1/sigma^2
    k_param = 0.7

    N_pos = 20
    N_neg = 20
    N_unl = 200

    foo = [co.matrix(1,(4,5),'i'), co.matrix(1,(4,5),'i'), co.matrix(1,(4,5),'i')]
    print(len(foo))

    # generate training labels
    yp = co.matrix(1,(1,N_pos),'i')
    yu = co.matrix(0,(1,N_unl),'i')
    yn = co.matrix(-1,(1,N_neg),'i')
    Dy = co.matrix([[yp], [yu], [yn], [yn], [yn], [yn]])

    # generate training data
    co.setseed(11)
    Dtrainp = co.normal(2,N_pos)*0.4
    Dtrainu = co.normal(2,N_unl)*0.5
    Dtrainn = co.normal(2,N_neg)*0.3
    Dtrain21 = Dtrainn-1
    Dtrain21[0,:] = Dtrainn[0,:]+1
    Dtrain22 = -Dtrain21

    # training data
    Dtrain = co.matrix([[Dtrainp], [Dtrainu], [Dtrainn+1.0], [Dtrainn-1.0], [Dtrain21], [Dtrain22]])
    Dtrain = np.array(Dtrain)

    # build the training kernel
    kernel = get_kernel(Dtrain, Dtrain, type=k_type, param=k_param)

    # use SSAD
    ssad = ConvexSSAD(kernel, Dy, 1.0, 1.0, 1.0 / (N_unl * 0.1), 1.0)
    ssad.fit()

    # build the test kernel
    kernel = get_kernel(Dtrain, Dtrain[:,ssad.get_support_dual()], type=k_type, param=k_param)

    thres = ssad.get_threshold()
    pred = ssad.apply(kernel)
    pred = np.array(pred)
    pred = pred.transpose()

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
    kernel = get_kernel(Dtest, Dtrain[:,ssad.get_support_dual()], type=k_type, param=k_param)
    res = ssad.apply(kernel)

    # make a nice plot of it
    Z = np.reshape(res,(sx,sy))
    plt.contourf(X, Y, Z, 20)
    plt.contour(X, Y, Z, [ssad.get_threshold()])
    plt.scatter(Dtrain[0,ssad.get_support_dual()],Dtrain[1,ssad.get_support_dual()],60,c='w')
    plt.scatter(Dtrain[0,N_pos:N_pos+N_unl-1],Dtrain[1,N_pos:N_pos+N_unl-1],10,c='g')
    plt.scatter(Dtrain[0,0:N_pos],Dtrain[1,0:N_pos],20,c='r')
    plt.scatter(Dtrain[0,N_pos+N_unl:],Dtrain[1,N_pos+N_unl:],20,c='b')

    plt.show()
    print('finished')