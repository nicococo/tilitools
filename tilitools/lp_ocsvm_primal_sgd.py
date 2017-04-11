import numpy as np
from functools import partial

from utils_optimize import optimize_subgradient_descent


class LpOcSvmPrimalSGD:
    """ Lp-norm regularized primal one-class support vector machine.

    """
    PRECISION = 1e-3 # important: effects the threshold, support vectors and speed!
    w = None     # (vector) parameter vector
    nu = 1.0	 # (scalar) the regularization constant 1/n <= nu <= 1
    pnorm = 2.0  # (scalar) p-norm
    threshold = 0.0	 # (scalar) the optimized threshold (rho)
    outliers = None  # (vector) indices of real outliers in the training sample

    def __init__(self, pnorm=2., nu=1.0):
        self.nu = nu
        self.pnorm = pnorm
        print('Creating new primal lp (p={0}) one-class svm with C=1/(n*nu) (nu={1}).'.format(pnorm, nu))

    def fit(self, X):
        # number of training examples
        x0 = np.zeros(X.shape[0]+1)
        x0[0] = np.max(np.var(X, axis=1))
        x0[1:] = np.mean(X, axis=1)

        xstar, obj, iter = optimize_subgradient_descent(x0,
                                                        partial(fun, data=X, p=self.pnorm, nu=self.nu),
                                                        partial(grad, data=X, p=self.pnorm, nu=self.nu),
                                                        1000, 1e-3, 0.01)
        print obj, iter
        print xstar

        self.threshold = xstar[0]
        self.w = xstar[1:]
        scores = self.apply(X)
        self.outliers = np.where(scores < 0.)[0]

    def get_threshold(self):
        return self.threshold

    def get_outliers(self):
        return self.outliers

    def apply(self, X):
        return self.w.T.dot(X) - self.threshold


def fun(x, data, p, nu):
    feats, n = data.shape
    w = x[1:]
    rho = x[0]
    pnorm = np.sum(np.abs(w)**p)**(1./p)
    slacks = rho - w.T.dot(data)
    slacks[slacks < 0.] = 0.
    return pnorm - rho + 1./(np.float(n)*nu) * np.sum(slacks)


def grad(x, data, p, nu):
    feats, n = data.shape
    C = 1./(np.float(n)*nu)
    w = x[1:]
    rho = x[0]
    pnorm1 = np.sum(np.abs(w)**p)**(1./(p-1.))
    grad_pnorm = (w*np.abs(w)**(p-2)) / pnorm1

    slacks = rho - w.T.dot(data)
    inds = np.where(slacks >= 0.)[0]

    grad = np.zeros(feats+1)
    grad[0] = -1. + C * np.float(inds.size)
    grad[1:] = grad_pnorm + C * np.sum(data[:, inds], axis=1)
    return grad
