from cvxopt import matrix
from cvxopt.lapack import syev
import numpy as np


class LatentPCA:
    """ Structured Extension for Principle Component Analysis.
        Written by Nico Goernitz, TU Berlin, 2014
    """

    def __init__(self, sobj):
        self.sobj = sobj    # structured object
        self.sol = None     # (vector) solution vector (after training, of course)
        self.latent = None

    def fit(self, max_iter=50):
        """ Solve the optimization problem with a
            sequential convex programming/DC-programming
            approach:
            Iteratively, find the most likely configuration of
            the latent variables and then, optimize for the
            model parameter using fixed latent states.
        """
        samples = self.sobj.get_num_samples()
        dims = self.sobj.get_num_dims()

        self.latent = np.random.randint(0, self.sobj.get_num_states(), samples)
        self.sol = np.random.randn(dims)
        psi = np.zeros((dims, samples))
        old_psi = np.zeros((dims, samples))
        threshold = 0.
        iter = 0
        # terminate if objective function value doesn't change much
        while iter < max_iter and (iter < 2 or np.sum(np.abs(psi-old_psi)) >= 0.001):
            print('Starting iteration {0}.'.format(iter))
            print(np.sum(np.abs(psi-old_psi)))
            iter += 1
            old_psi = psi.copy()

            # 1. linearize
            # for the current solution compute the
            # most likely latent variable configuration
            mean = np.zeros(dims)
            for i in range(samples):
                _, self.latent[i], psi[:, i] = self.sobj.argmax(self.sol, i)
                mean += psi[:, i]
            mean /= np.float(samples)
            mpsi = psi - mean.reshape((dims, 1))

            # 2. solve the intermediate convex optimization problem
            A = mpsi.dot(mpsi.T)
            W = np.zeros((dims, dims))
            syev(matrix(A), matrix(W), jobz='V')
            self.sol = np.array(A[:, dims-1]).ravel()
        return self.sol, self.latent, threshold

    def apply(self, pred_sobj):
        """ Application of the StructuredPCA:

            score = max_z <sol*,\Psi(x,z)>
            latent_state = argmax_z <sol*,\Psi(x,z)>
        """
        samples = pred_sobj.get_num_samples()
        vals = np.zeros(samples)
        structs = []
        for i in range(samples):
            vals[i], struct, _ = pred_sobj.argmax(self.sol, i)
            structs.append(struct)
        return vals, structs
