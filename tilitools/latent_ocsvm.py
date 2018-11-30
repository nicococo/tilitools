import numpy as np

from tilitools.utils_kernel import get_kernel, center_kernel, normalize_kernel
from tilitools.ocsvm_dual_qp import OcSvmDualQP


class LatentOCSVM:
    """ Structured One-class SVM (a.k.a Structured Anomaly Detection).
        Written by Nico Goernitz, TU Berlin, 2014
    """
    nu = 1.0	 # (scalar) the regularization constant > 0
    sobj = None  # structured object contains various functions
                 # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)
    sol = None   # (vector) solution vector (after training, of course)
    slacks = None
    svs_inds = None
    threshold = 0.0
    mean_psi = None
    norm_ord = 1

    def __init__(self, sobj, nu=1.0, norm_ord=1):
        self.nu = nu
        self.sobj = sobj
        self.norm_ord = norm_ord

    def fit(self, max_iter=50, hotstart=None, prec=1e-3, center=False, normalize=False):
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

        sol = self.sobj.get_hotstart_sol()
        if hotstart is not None and hotstart.size == DIMS:
            print('New hotstart position defined.')
            sol = hotstart

        psi = np.zeros((DIMS, N))  # (dim x exm)
        old_psi = np.zeros((DIMS, N))  # (dim x exm)
        threshold = 0.

        # terminate if objective function value doesn't change much
        iter = 0
        allobjs = []
        while iter < max_iter and (iter < 2 or np.sum(abs(np.array(psi-old_psi))) >= prec):
            print('Starting iteration {0}.'.format(iter))
            print(np.sum(abs(np.array(psi-old_psi))))
            iter += 1
            old_psi = psi.copy()

            # 1. most likely configuration
            # for the current solution compute the
            # most likely latent variable configuration
            for i in range(N):
                _, latent[i], psi[:, i] = self.sobj.argmax(sol, i)
                psi[:, i] /= np.linalg.norm(psi[:, i], ord=self.norm_ord)

            # 2. solve the intermediate convex optimization problem
            kernel = get_kernel(psi, psi)
            if center:
                kernel = center_kernel(kernel)
            if normalize:
                kernel = normalize_kernel(kernel)
            svm = OcSvmDualQP(kernel, self.nu)
            svm.fit()
            threshold = svm.get_threshold()

            self.svs_inds = svm.get_support_dual()
            sol = psi.dot(svm.get_alphas())

            # calculate objective
            self.threshold = threshold
            slacks = threshold - sol.T.dot(psi)
            slacks[slacks < 0.0] = 0.0
            obj = 0.5*sol.T.dot(sol) - threshold + 1./(np.float(N)*self.nu) * np.sum(slacks)
            print("Iter {0}: Values (Threshold-Slacks-Objective) = {1}-{2}-{3}".format(
                iter, threshold, np.sum(slacks), obj))
            allobjs.append(obj)
        self.slacks = slacks
        self.sol = sol
        self.latent = latent
        return sol, latent, threshold

    def apply(self, pred_sobj):
        """ Application of the StructuredOCSVM:
            anomaly_score = max_z <sol*,\Psi(x,z)>
            latent_state = argmax_z <sol*,\Psi(x,z)>
        """
        N = pred_sobj.get_num_samples()
        vals = np.zeros(N)
        structs = []
        for i in range(N):
            vals[i], struct, psi = pred_sobj.argmax(self.sol, i)
            vals[i] = (vals[i]/np.linalg.norm(psi, ord=self.norm_ord) - self.threshold)
            structs.append(struct)
        return vals, structs
