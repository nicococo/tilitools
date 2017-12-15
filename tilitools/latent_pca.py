from cvxopt import matrix,spmatrix,sparse,uniform,normal,setseed
from cvxopt.lapack import syev
import numpy as np

class LatentPCA:
    """ Structured Extension for Principle Component Analysis.
        Written by Nico Goernitz, TU Berlin, 2014
    """
    sobj = None  # structured object contains various functions
              # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)
    sol = None  # (vector) solution vector (after training, of course)

    def __init__(self, sobj):
        self.sobj = sobj

    def fit(self, max_iter=50):
        """ Solve the optimization problem with a
            sequential convex programming/DC-programming
            approach:
            Iteratively, find the most likely configuration of
            the latent variables and then, optimize for the
            model parameter using fixed latent states.
        """
        N = self.sobj.get_num_samples()
        DIMS = self.sobj.get_num_dims()

        # intermediate solutions
        # latent variables
        latent = [0.0]*N

        sol = np.random.randn(DIMS)
        psi = np.zeros((DIMS, N)) # (dim x exm)
        old_psi = np.zeros((DIMS,N)) # (dim x exm)
        threshold = 0.
        obj = -1.
        iter = 0
        # terminate if objective function value doesn't change much
        while iter < max_iter and (iter < 2 or np.sum(abs(np.array(psi-old_psi))) >= 0.001):
            print('Starting iteration {0}.'.format(iter))
            print(np.sum(abs(np.array(psi-old_psi))))
            iter += 1
            old_psi = psi.copy()

            # 1. linearize
            # for the current solution compute the
            # most likely latent variable configuration
            mean = np.zeros(DIMS)
            for i in range(N):
                _, latent[i], psi[:, i] = self.sobj.argmax(sol, i)
                mean += psi[:, i]
            mean /= float(N)
            mpsi = psi - np.repeat(mean.reshape((DIMS, 1)), N, axis=1)

            # 2. solve the intermediate convex optimization problem
            A = mpsi.dot(mpsi.T)
            W = np.zeros((DIMS, DIMS))
            syev(matrix(A), matrix(W), jobz='V')
            sol = np.array(A[:, DIMS-1]).reshape(DIMS)

        print(np.sum(abs(np.array(psi-old_psi))))
        self.sol = sol
        self.latent = latent
        return sol, latent, threshold

    def apply(self, pred_sobj):
        """ Application of the StructuredPCA:

            score = max_z <sol*,\Psi(x,z)>
            latent_state = argmax_z <sol*,\Psi(x,z)>
        """
        N = pred_sobj.get_num_samples()
        vals = np.zeros(N)
        structs = []
        for i in range(N):
            vals[i], struct, _ = pred_sobj.argmax(self.sol, i)
            structs.append(struct)
        return vals, structs
