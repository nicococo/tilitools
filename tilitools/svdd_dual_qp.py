from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp
import numpy as np

from utils_kernel import get_diag_kernel, get_kernel


class SvddDualQP:
    """ Dual QP implementation of the support vector data description (SVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """
    PRECISION = 1e-3  # important: effects the threshold, support vectors and speed!

    kernel = None 	# (string) name of the kernel to use
    kparam = None 	# (-) kernel parameter
    samples = -1 	# (scalar) amount of training data in X

    nu = 0.95	    # (scalar) the regularization constant > 0

    X = None        # (matrix) training data
    alphas = None   # (vector) dual solution vector
    svs = None      # (vector) support vector indices
    radius2 = 0.0   # (scalar) the optimized threshold (rho)
    cTc = None      # (vector) alphaT*K*alpha for support vectors only

    pobj = 0.0      # (scalar) primal objective value after training

    def __init__(self, kernel, kparam, nu):
        self.kernel = kernel
        self.kparam = kparam
        self.nu = nu
        print('Creating new dual QP SVDD ({0}) with nu={1}.'.format(kernel, nu))

    def fit(self, X, max_iter=-1):
        """
        :param X: Data matrix is assumed to be feats x samples.
        :param max_iter: *ignored*, just for compatibility.
        :return: Alphas and threshold for dual SVDDs.
        """
        self.X = X.copy()
        dims, self.samples = X.shape
        if self.samples < 1:
            print('Invalid training data.')
            return -1

        # number of training examples
        N = self.samples

        kernel = get_kernel(X, X, self.kernel, self.kparam)
        norms = np.diag(kernel).copy()

        if self.nu >= 1.0:
            print("Center-of-mass solution.")
            self.alphas = np.ones(self.samples) / float(self.samples)
            self.radius2 = 0.0
            self.svs = np.array(range(self.samples), dtype='i')
            self.pobj = 0.0  # TODO: calculate real primal objective
            self.cTc = self.alphas[self.svs].T.dot(kernel[self.svs, :][:, self.svs].dot(self.alphas[self.svs]))
            return self.alphas, self.radius2

        C = 1. / np.float(self.samples*self.nu)

        # generate a kernel matrix
        P = 2.0*matrix(kernel)

        # this is the diagonal of the kernel matrix
        q = -matrix(norms)

        # sum_i alpha_i = A alpha = b = 1.0
        A = matrix(1.0, (1, N))
        b = matrix(1.0, (1, 1))

        # 0 <= alpha_i <= h = C
        G1 = spmatrix(1.0, range(N), range(N))
        G = sparse([G1, -G1])
        h1 = matrix(C, (N, 1))
        h2 = matrix(0.0, (N, 1))
        h = matrix([h1, h2])

        sol = qp(P, q, G, h, A, b)

        # store solution
        self.alphas = np.array(sol['x'], dtype=np.float)
        self.pobj = -sol['primal objective']

        # find support vectors
        self.svs = np.where(self.alphas > self.PRECISION)[0]
        # self.cTc = self.alphas[self.svs].T.dot(kernel[self.svs, :][:, self.svs].dot(self.alphas[self.svs]))
        self.cTc = self.alphas.T.dot(kernel.dot(self.alphas))

        # find support vectors with alpha < C for threshold calculation
        self.radius2 = 0.
        thres = self.predict(X[:, self.svs])
        self.radius2 = np.min(thres)
        return self.alphas, thres

    def get_radius(self):
        return self.radius2

    def get_alphas(self):
        return self.alphas

    def get_support_inds(self):
        return self.svs

    def get_support(self):
        return self.alphas[self.svs]

    def predict(self, Y):
        # build test kernel
        kernel = get_kernel(Y, self.X[:, self.svs], self.kernel, self.kparam)
        # kernel = Kernel.get_kernel(Y, self.X, self.kernel, self.kparam)
        # for svdd we need the data norms additionally
        norms = get_diag_kernel(Y, self.kernel)
        # number of training examples
        res = self.cTc - 2. * kernel.dot(self.get_support()).T + norms
        # res = self.cTc - 2. * kernel.dot(self.alphas).T + norms
        return res.reshape(Y.shape[1]) - self.radius2