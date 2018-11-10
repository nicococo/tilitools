import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
import sys

from tilitools.utils import profile


class BDD:
    """
    Bayesian data description
    """
    is_trained = False

    def __init__(self, kernel, nu=0.5):
        self.kernel = np.array(kernel)
        self.nu = nu
        self.n = self.kernel.shape[1]
        self.diagonal = np.diag(np.sum(self.kernel, axis=1))
        self.cov_mat = np.eye(self.n)
        print('Creating new SVDD with {0} samples.'.format(self.n))

    @profile
    def fit(self):
        """
        train a BDD
        """
        # number of training samples
        n = self.n
        # the kernel matrix
        K = self.kernel
        # the covariance matrix
        C = self.cov_mat
        # the inverse of the covariance matrix
        C_inv = np.linalg.inv(C)
        # the diagonal matrix containing the sum of the rowa of the kernel matrix
        D = self.diagonal
        # parameter 0 < nu < 1, controling the sparsity of the solution
        nu = self.nu
        # the mean vector
        m = -np.dot(D, np.ones(n))**nu
        # solve the quadratic program
        P = matrix(n*K + C_inv)
        q = matrix(-1. * (np.dot(D, np.ones(n)) + np.dot(C_inv, m)).T)
        sol = qp(P, q)
        # extract the solution
        self.alphas = sol['x']
        # BDD is trained
        self.is_trained = True


    def apply(self, test_data, kernel_map, norms):
        """
        apply the trained BDD
            test_data - the matrix of the test data
            kernel_map - the matrix of the kernel map
            norms - the diagonal of the test kernel

            return:
                scores - the scores for each data point
                sorted_data - the data sorted by their scores

        """
        # check if BDD is trained
        if not self.is_trained:
            print('First train, then test.')
            sys.exit()
        # the kernel map
        K_map = np.array(kernel_map)
        # the diagonal of the test kernel
        norms = np.array(norms)
        # the alphas
        alphas = np.array(self.alphas)
        n_test = K_map.shape[0]
        X = test_data
        # compute the scores
        K_sum = np.dot(np.dot(alphas.T, self.kernel),alphas)
        scores = (K_sum + norms - 2.*np.sum(K_map,axis=1))[0]
        # sort the test data by their scores
        sort_indices = np.argsort(scores)
        sorted_data =  matrix(X[:, sort_indices])
        return scores, sorted_data