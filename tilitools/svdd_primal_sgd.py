__author__ = 'nicococo'
import numpy as np

from numba import autojit


class SvddPrimalSGD(object):
    """ Primal subgradient descent solver for the support vector data description (SVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """
    PRECISION = 10**-3  # important: effects the threshold, support vectors and speed!
    nu = 0.95	    # (scalar) the regularization constant > 0
    c = None        # (vecor) center of the hypersphere
    radius2 = 0.0   # (scalar) the optimized threshold (rho)
    pobj = 0.0      # (scalar) primal objective after training

    def __init__(self, nu):
        self.nu = nu
        print('Creating new primal SVDD with nu={0}.'.format(nu))

    @autojit
    def fit(self, X, max_iter=20000, prec=1e-6, rate=0.01):
        if X.shape[1] < 1:
            print('Invalid training data.')
            return -1, -1
        self.c, self.radius2, self.pobj, iter = fit_extern(X, self.nu, max_iter, prec, rate)
        print('Iter={2}: obj={0}  T={1}'.format(self.pobj, self.radius2, iter+1))
        return self.c, self.radius2

    def get_radius(self):
        return self.radius2

    def predict(self, X):
        # X : (dims x samples)
        dist = self.c.T.dot(self.c) - 2.*self.c.T.dot(X) + np.sum(X*X, axis=0)
        return dist - self.radius2


@autojit(nopython=True)
def fit_extern(X, nu, max_iter, prec, rate):
    """ Subgradient descent solver for primal SVDD.
        Optimized for 'numba'
    """
    (dims, samples) = X.shape

    # number of training examples
    reg = 1./(np.float64(samples)*nu)

    # center of mass
    c = np.zeros(dims, dtype=np.float64)
    # np.sum(X*X, axis=0)
    sum_XX = np.zeros(samples)
    for s in range(samples):
        foo = 0.0
        for d in range(dims):
            foo += X[d, s]*X[d, s]
            c[d] += X[d, s] / np.float64(samples)
        sum_XX[s] = foo
    # print np.sum(np.abs(c-np.mean(X, axis=1)))

    dot_2cX = np.zeros(samples, dtype=np.float64)
    for s in range(samples):
        dot_2cX[s] = 2.0 * np.sum(c*X[:, s])
    dist = np.sum(c*c) - dot_2cX + sum_XX

    T = 0.4 * np.max(dist) * (1.0-nu)  # starting heuristic T
    # if nu exceeds 1.0, then T^* is always 0 and c can
    # be computed analytically (as center-of-mass, mean)
    if nu >= 1.0:
        return c, 0.0, 0.0, 0

    is_converged = False
    best_c = c
    best_radius2 = T
    obj_best = np.float64(1e20)

    obj_bak = -100.
    iter = 0

    # gradient step for center
    dc = np.zeros(dims, dtype=np.float64)
    inds = np.zeros(samples, dtype=np.int64)
    while not is_converged and iter < max_iter:
        # print iter
        for s in range(samples):
           dot_2cX[s] = 2.0 * np.sum(c*X[:, s])

        # calculate the distances of the center to each datapoint
        dist = np.sum(c*c) - dot_2cX + sum_XX
        inds_size = 0
        for s in range(samples):
            if dist[s]-T >= 1e-12:
                inds[inds_size] = s
                inds_size += 1
        # we need at least 1 entry, hence lower T to the maximum entry
        if inds_size == 0:
            inds_size = 1
            inds[0] = np.argmax(dist)
            T = dist[inds[0]]

        # real objective value given the current center c and threshold T
        ds = 0.0
        for s in range(inds_size):
            ds += dist[inds[s]] - T
        obj = T + reg*ds

        # this is subgradient, hence need to store the best solution so far
        if obj_best >= obj:
            best_c = c
            best_radius2 = T
            obj_best = obj

        # stop, if progress is too slow
        if obj > 0.:
            if np.abs((obj-obj_bak)/obj) < prec:
                is_converged = True
                continue
        obj_bak = obj

        # stepsize should be not more than 0.1 % of the maximum value encountered in dist
        max_change = rate * np.max(dist) / np.float(iter+1)*10.

        # gradient step for threshold
        dT = 1.0 - reg*np.float(inds_size)
        T -= np.sign(dT) * max_change

        # gradient step for center
        norm_dc = 0.0
        for d in range(dims):
            dc[d] = 0.0
            for s in range(inds_size):
                dc[d] += 2.*reg*(c[d] - X[d, inds[s]])
            norm_dc += dc[d]*dc[d]
        norm_dc = np.sqrt(norm_dc)

        if np.abs(norm_dc) < 1e-12:
            norm_dc = 1.0

        for d in range(dims):
            c[d] -= dc[d]/norm_dc * max_change
        iter += 1

    return best_c, best_radius2, obj_best, iter