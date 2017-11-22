from cvxopt import matrix, spmatrix, sparse, spdiag
from cvxopt.solvers import qp
import numpy as np

from utils import profile

class ConvexSSAD:
    """ Convex semi-supervised anomaly detection with hinge-loss and L2 regularizer
        as described in Goernitz et al., Towards Supervised Anomaly Detection, JAIR, 2013

               minimize 			0.5 ||w||^2_2 - rho - kappa*gamma + eta_u sum_i xi_i + eta_l sum_j xi_j
        {w,rho,gamma>=0,xi>=0}
        subject to   <w,phi(x_i)> >= rho - xi_i
                  y_j<w,phi(x_j)> >= y_j*rho + gamma - xi_j

        And the corresponding dual optimization problem:

            maximize -0.5 sum_(i,j) alpha_i alpha_j y_i y_j k(x_i,x_j)
        {0<=alpha_i<=eta_i}
            subject to 	kappa <= sum_j alpha_j  (for all labeled examples)
                        1 = sum_j y_i alpha_j  (for all examples)

        We introduce labels y_i = +1 for all unlabeled examples which enables us to combine sums.

        Note: Only dual solution is supported.

        Written by: Nico Goernitz, TU Berlin, 2013/14
    """
    PRECISION = 1e-6  # important: effects the threshold, support vectors and speed!

    cy = None  # (vector) converted label vector (+1 for pos and unlabeled, -1 for outliers)
    cl = None  # (vector) converted label vector (+1 for labeled examples, 0.0 for unlabeled)

    samples = -1 	# (scalar) amount of training data in X
    labeled = -1 	# (scalar) amount of labeled data

    cC = None   # (vector) converted upper bound box constraint for each example
    Cp = 1.0    # (scalar) the regularization constant for positively labeled samples > 0
    Cu = 1.0    # (scalar) the regularization constant for unlabeled samples > 0
    Cn = 1.0    # (scalar) the regularization constant for outliers > 0

    kappa = 1.0  # (scalar) regularizer for importance of the margin

    kernel = None  # (matrix) kernel matrix
    y = None  # (vector) corresponding labels (+1,-1 and 0 for unlabeled)

    alphas = None  # (vector) dual solution vector
    svs = None  # (vector) list of support vector (contains indices)

    threshold = 0.0 # (scalar) the optimized threshold (rho)

    def __init__(self, kernel, y, kappa=1.0, Cp=1.0, Cu=1.0, Cn=1.0):
        print('Check if labels y are only 1-d.')
        assert(len(y.shape) == 1)
        self.kernel = kernel
        self.y = y
        self.kappa = kappa
        self.Cp = Cp
        self.Cu = Cu
        self.Cn = Cn
        self.samples = y.size

        self.labeled = np.sum(np.abs(y))

        self.cy = y.copy().reshape((y.size, 1))
        self.cy[y == 0] = 1  # cy=+1.0 (unlabeled,pos) & cy=-1.0 (neg)
        self.cl = np.abs(y.copy())  # cl=+1.0 (labeled) cl=0.0 (unlabeled)

        self.cC = np.zeros(y.size) # cC=Cu (unlabeled) cC=Cp (pos) cC=Cn (neg)
        self.cC[y == 0] = Cu
        self.cC[y == 1] = Cp
        self.cC[y ==-1] = Cn

        # if there are no labeled examples, then set kappa to 0.0 otherwise
        # the dual constraint kappa <= sum_{i \in labeled} alpha_i = 0.0 will
        # prohibit a solution
        if self.labeled == 0:
            print('There are no labeled examples hence, setting kappa=0.0')
            self.kappa = 0.0
        print('Convex semi-supervised anomaly detection with {0} samples ({1} labeled).'.format(self.samples, self.labeled))

    def set_train_kernel(self, kernel):
        dim1, dim2 = kernel.shape
        print([dim1, dim2])
        assert(dim1 == dim2 and dim1 == self.samples)
        self.kernel = kernel

    @profile
    def fit(self, check_psd_eigs=False):
        # number of training examples
        N = self.samples

        # generate the label kernel
        Y = self.cy.dot(self.cy.T)

        # generate the final PDS kernel
        P = matrix(self.kernel*Y)

        # check for PSD
        if check_psd_eigs:
            eigs = np.linalg.eigvalsh(np.array(P))
            if eigs[0] < 0.0:
                print('Smallest eigenvalue is {0}'.format(eigs[0]))
                P += spdiag([-eigs[0] for i in range(N)])

        # there is no linear part of the objective
        q = matrix(0.0, (N, 1))

        # sum_i y_i alpha_i = A alpha = b = 1.0
        A = matrix(self.cy, (1, self.samples), 'd')
        b = matrix(1.0, (1, 1))

        # inequality constraints: G alpha <= h
        # 1) alpha_i  <= C_i
        # 2) -alpha_i <= 0
        G12 = spmatrix(1.0, range(N), range(N))
        h1 = matrix(self.cC)
        h2 = matrix(0.0, (N, 1))
        G = sparse([G12, -G12])
        h = matrix([h1, h2])
        if self.labeled > 0:
            # 3) kappa <= \sum_i labeled_i alpha_i -> -cl' alpha <= -kappa
            print('Labeled data found.')
            G3 = -matrix(self.cl, (1, self.cl.size), 'd')
            h3 = -matrix(self.kappa, (1, 1))
            G = sparse([G12, -G12, G3])
            h = matrix([h1, h2, h3])

        # solve the quadratic programm
        sol = qp(P, -q, G, h, A, b)

        # store solution
        self.alphas = np.array(sol['x'])

        # 1. find all support vectors, i.e. 0 < alpha_i <= C
        # 2. store all support vector with alpha_i < C in 'margins'
        self.svs = np.where(self.alphas >= ConvexSSAD.PRECISION)[0]

        # these should sum to one
        print('Validate solution:')
        print('- found {0} support vectors'.format(len(self.svs)))
        print('0 <= alpha_i : {0} of {1}'.format(np.sum(0. <= self.alphas), N))
        print('- sum_(i) alpha_i cy_i = {0} = 1.0'.format(np.sum(self.alphas*self.cy)))
        print('- sum_(i in sv) alpha_i cy_i = {0} ~ 1.0 (approx error)'.format(np.sum(self.alphas[self.svs]*self.cy[self.svs])))
        print('- sum_(i in labeled) alpha_i = {0} >= {1} = kappa'.format(np.sum(self.alphas[self.cl == 1]), self.kappa))
        print('- sum_(i in unlabeled) alpha_i = {0}'.format(np.sum(self.alphas[self.y == 0])))
        print('- sum_(i in positives) alpha_i = {0}'.format(np.sum(self.alphas[self.y == 1])))
        print('- sum_(i in negatives) alpha_i = {0}'.format(np.sum(self.alphas[self.y ==-1])))

        # infer threshold (rho)
        psvs = np.where(self.y[self.svs] == 0)[0]
        # case 1: unlabeled support vectors available
        if psvs.size > 0:
            psvs = self.svs[psvs]
            k = self.kernel[:, self.svs]
            k = k[psvs, :]
            thres = self.apply(k)
            inds = np.where(self.alphas[psvs].reshape(psvs.size) <= self.Cu-ConvexSSAD.PRECISION)[0]
            if inds.size > 0:
                self.threshold = np.min(thres[inds])
            else:
                # if no alpha < 1.-precision could be found
                self.threshold = np.max(thres)
                # still, this can be wrong if all unsupervised SVs are outliers
            self.outliers = self.svs[np.where(thres < self.threshold)[0]]
        else:
        # case 2: only labeled examples available
            k = self.kernel[:, self.svs]
            k = k[self.svs, :]
            thres = self.apply(k)
            pinds = np.where(self.y[self.svs] == +1)[0]
            ninds = np.where(self.y[self.svs] == -1)[0]
            # only negatives is not possible
            if ninds.size > 0 and pinds.size == 0:
                print('ERROR: Check pre-defined PRECISION.')
                self.threshold = np.max(thres[ninds])
            elif ninds.size == 0:
                self.threshold = np.min(thres[pinds])
            else:
                # smallest negative + largest positive
                p = np.max(thres[pinds])
                n = np.min(thres[ninds])
                self.threshold = (n+p)/2.
            self.outliers = self.svs[np.where(thres < self.threshold)[0]]
        print('Found {0} support vectors. {1} of them are outliers.'.format(len(self.svs), self.outliers))

    def get_threshold(self):
        return self.threshold

    def get_support_dual(self):
        return self.svs

    def get_alphas(self):
        return self.alphas

    def apply(self, kernel):
        """ Application of dual trained ssad.
            kernel = get_kernel(Y, X[:, cssad.svs], kernel_type, kernel_param)
        """
        inds = self.svs
        if kernel.shape[1] == self.samples:
            # if kernel is not restricted to support vectors
            inds = np.arange(self.samples)
        ay = self.alphas[inds] * self.cy[inds]
        return ay.T.dot(kernel.T).T - self.threshold
