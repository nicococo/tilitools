import numpy as np
from numba import jit
from functools import partial
from scipy.optimize import minimize

from tilitools.profiler import profile


class HuberOcsvmPrimal:
    """ Smooth Huber-loss primal one-class support vector machine.
    """
    w = None     # (vector) parameter vector
    nu = 1.0	 # (scalar) the regularization constant 1/n <= nu <= 1
    threshold = 0.0	 # (scalar) the optimized threshold (rho)
    outliers = None  # (vector) indices of real outliers in the training sample

    def __init__(self,  nu=1.0):
        self.nu = nu

    @profile
    def fit(self, X, max_iter=1000, prec=1e-3, verbosity=0):
        # number of training examples
        feats, n = X.shape
        x0 = np.zeros(feats+1)
        x0[1:] = np.mean(X, axis=1)
        x0[1:] /= np.linalg.norm(x0[1:])
        x0[0] = np.linalg.norm(np.mean(X, axis=1))
        if verbosity > 0:
            print('Threshold is {0}'.format(x0[0]))
            print('Norm of w is {0}'.format(np.linalg.norm(x0[1:])))

        fun = partial(fun_smooth_ocsvm, X=X, nu=self.nu, delta=0., epsilon=0.5)
        grad = partial(grad_smooth_ocsvm, X=X, nu=self.nu, delta=0., epsilon=0.5)

        res = minimize(fun, x0, jac=grad, method='L-BFGS-B',
                       options={'gtol': prec, 'disp': verbosity > 0, 'maxiter' : max_iter})
        self.w = res.x[1:]
        self.threshold = res.x[0]
        scores = self.apply(X)
        self.outliers = np.where(scores < 0.)[0]
        if verbosity > 0:
            print('---------------------------------------------------------------')
            print('Stats:')
            print('Number of samples: {0}, nu: {1}; C: ~{2:1.2f}; %Outliers: {3:3.2f}%.'
                  .format(n, self.nu, 1./(self.nu*n),np.float(self.outliers.size) / np.float(n) * 100.0))
            print('Threshold is {0}'.format(self.threshold))
            print('Norm of w is {0}'.format(np.linalg.norm(self.w)))
            print('Iterations {0}'.format(iter))
            print('---------------------------------------------------------------')

    def get_threshold(self):
        return self.threshold

    def get_outliers(self):
        return self.outliers

    def apply(self, X):
        return self.w.T.dot(X) - self.threshold


def fun_smooth_ocsvm(var, X, nu, delta, epsilon):
    rho = var[0]
    w = var[1:]
    w = w.reshape(w.size, 1)

    n = X.shape[1]
    d = X.shape[0]

    inner = (rho - w.T.dot(X)).ravel()
    loss = np.zeros(n)

    inds = np.argwhere(inner >= delta + epsilon)
    loss[inds] = inner[inds] - delta

    inds = np.argwhere(np.logical_and((delta - epsilon <= inner), (inner <= delta + epsilon))).ravel()
    loss[inds] = (epsilon + inner[inds] - delta) * (epsilon + inner[inds] - delta) / (4. * epsilon)

    f = 1. / 2. * w.T.dot(w) - rho + np.sum(loss) / (n * nu)
    return f[0, 0]


def grad_smooth_ocsvm(var, X, nu, delta, epsilon):
    rho = var[0]
    w = var[1:]
    w = w.reshape(w.size, 1)

    n = X.shape[1]
    d = X.shape[0]

    inner = (rho - w.T.dot(X)).ravel()
    grad_loss_rho = np.zeros(n)
    grad_loss_w = np.zeros((n, d))

    inds = np.argwhere(inner >= delta + epsilon).ravel()
    grad_loss_rho[inds] = 1.
    grad_loss_w[inds, :] = -X[:, inds].T

    inds = np.argwhere(np.logical_and((delta - epsilon <= inner), (inner <= delta + epsilon))).ravel()
    grad_loss_rho[inds] = (-delta + epsilon + inner[inds]) / (2. * epsilon)
    grad_loss_w[inds, :] = ((-delta + epsilon + inner[inds]) / (2. * epsilon) * (-X[:, inds])).T

    grad = np.zeros(d + 1)
    grad[0] = -1 + np.sum(grad_loss_rho) / (n * nu)
    grad[1:] = w.ravel() + np.sum(grad_loss_w, axis=0) / (n * nu)
    return grad.ravel()
