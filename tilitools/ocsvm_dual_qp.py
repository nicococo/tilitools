from cvxopt import matrix,spmatrix,sparse
from cvxopt.solvers import qp
import numpy as np

from utils import profile


class OcSvmDualQP:
    """ One-class support vector machine

        'Estimating the support of a high-dimensional distribution.',
        Sch\"{o}lkopf, B and Platt, J C and Shawe-Taylor, J and Smola, a J and Williamson, R C, Microsoft, 1999

    """
    PRECISION = 1e-3 # important: effects the threshold, support vectors and speed!

    kernel = None 	# (matrix) our training kernel n x n
    nu = 1.0	 # (scalar) the regularization constant 1/n <= nu <= 1

    alphas = None  # (vector) dual solution vector
    svs = None  # (vector) support vector indices
    threshold = 0.0	 # (scalar) the optimized threshold (rho)
    outliers = None  # (vector) indices of real outliers in the training sample

    def __init__(self, kernel, nu=1.0):
        self.kernel = kernel
        self.nu = nu
        samples, _ = kernel.shape
        print('Creating new one-class svm with {0} samples and C=1/(n*nu)={1}.'.format(samples, 1./(samples * nu)))

    @profile
    def fit(self):
        """ Trains an one-class svm in dual with kernel. """
        # number of training examples
        N = self.kernel.shape[0]

        # generate a kernel matrix
        P = matrix(self.kernel)
        # there is no linear part of the objective
        q = matrix(0.0, (N, 1))

        # Re-scaled variant:
        #   0 <= alpha <= 1
        #   sum_i alpha_i = n*nu
        # Instead of the original:
        #   0 <= alpha <= 1/n*nu
        #   sum_i alpha_i = 1

        # sum_i alpha_i = A alpha = b = 1.0
        A = matrix(1.0, (1, N))
        # b = matrix(1.0, (1, 1))
        b = matrix(N*self.nu, (1, 1))  # re-scaled variant

        # 0 <= alpha_i <= h = C
        G1 = spmatrix(1.0, range(N), range(N))
        G = sparse([G1,-G1])
        # h1 = matrix(1. / (N*self.nu), (N, 1))
        h1 = matrix(1., (N, 1))  # re-scaled variant
        h2 = matrix(0.0, (N, 1))
        h = matrix([h1, h2])
        sol = qp(P, -q, G, h, A, b)

        # store solution
        self.alphas = np.array(sol['x']).reshape((N, 1))
        # find support vectors
        self.svs = np.where(self.alphas >= OcSvmDualQP.PRECISION)[0]

        k = self.kernel[:, self.svs]
        k = k[self.svs, :]
        thres = self.apply(k)
        inds = np.where(self.alphas[self.svs] <= 1. - OcSvmDualQP.PRECISION)[0]
        if inds.size > 0:
            self.threshold = np.min(thres[inds])
        else:
            # if no alpha < 1.-precision could be found
            self.threshold = np.max(thres)

        self.outliers = self.svs[np.where(thres < self.threshold)[0]]
        print('---------------------------------------------------------------')
        print('Stats:')
        print('Number of samples: {0}, nu: {1}; C: ~{2:1.2f}; %SVs: {3:3.2f}%; %Outlierss: {4:3.2f}%.'
              .format(N, self.nu, 1./(self.nu*N), np.float(self.svs.size) / np.float(N) * 100.0,
              np.float(np.sum(thres < self.threshold)) / np.float(N) * 100.0))
        print('Hyperparameter nu ({0}) is an upper bound on the fraction of outliers ({0} >= {1:3.2f}%). '
              .format(self.nu, np.float(np.sum(thres < self.threshold)) / np.float(N) * 100.0))
        print('Hyperparameter nu ({0}) is a lower bound on the fraction of SVs ({0} <= {1:3.2f}%). '
              .format(self.nu, np.float(self.svs.size) / np.float(N) * 100.0))
        print('Sum of alphas ({0}) and deviation from n*nu ({1}).'.format(
            np.sum(self.alphas), np.abs(N*self.nu-np.sum(self.alphas))))
        print('Sum of SV alphas ({0}) and deviation from n*nu ({1}).'.format(
            np.sum(self.alphas[self.svs]), np.abs(N*self.nu-np.sum(self.alphas[self.svs]))))
        print('Number of SVs ({0}) of ({1}) total datapoints.'.format(
            self.svs.size, N))
        print('Threshold is {0}'.format(self.threshold))
        print('Objective is {0}'.format(-0.5*self.alphas.T.dot(self.kernel).dot(self.alphas)))
        print('---------------------------------------------------------------')

    def get_threshold(self):
        return self.threshold

    def get_support_dual(self):
        return self.svs

    def get_outliers(self):
        return self.outliers

    def get_alphas(self):
        return self.alphas

    def get_support_dual_values(self):
        return self.alphas[self.svs]

    def set_train_kernel(self, kernel):
        dim1, dim2 = kernel.shape
        if dim1!=dim2 and dim1!=self.kernel.shape[0]:
            print('(Kernel) Wrong format.')
            return
        self.kernel = kernel

    def apply(self, kernel):
        return kernel.dot(self.alphas[self.svs]) - self.threshold
