from cvxopt import matrix, normal
import numpy as np

from utils_kernel import get_kernel
from svdd_dual_qp import SvddDualQP

class LatentSVDD:
    """ Latent variable support vector data description.
        Written by Nico Goernitz, TU Berlin, 2014

        For more information see:
        'Learning and Evaluation with non-i.i.d Label Noise'
        Goernitz et al., AISTATS & JMLR W&CP, 2014
    """
    PRECISION = 10**-3 # important: effects the threshold, support vectors and speed!

    C = 1.0	# (scalar) the regularization constant > 0
    sobj = [] # structured object contains various functions
              # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)
    sol = [] # (vector) solution vector (after training, of course)


    def __init__(self, sobj, C=1.0):
        self.C = C
        self.sobj = sobj

    def train(self, max_iter=50):
        """ Solve the LatentSVDD optimization problem with a
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
        latent = [0]*N

        sol = 10.0*normal(DIMS,1)
        psi = matrix(0.0, (DIMS,N)) # (dim x exm)
        old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)
        threshold = 0

        obj = -1
        iter = 0

        # terminate if objective function value doesn't change much
        while iter<max_iter and (iter<2 or sum(sum(abs(np.array(psi-old_psi))))>=0.001):
            print('Starting iteration {0}.'.format(iter))
            print(sum(sum(abs(np.array(psi-old_psi)))))
            iter += 1
            old_psi = matrix(psi)

            # 1. linearize
            # for the current solution compute the
            # most likely latent variable configuration
            for i in range(N):
                # min_z ||sol - Psi(x,z)||^2 = ||sol||^2 + min_z -2<sol,Psi(x,z)> + ||Psi(x,z)||^2
                # Hence => ||sol||^2 - max_z  2<sol,Psi(x,z)> - ||Psi(x,z)||^2
                foo, latent[i], psi[:,i] = self.sobj.argmax(sol, i, opt_type='quadratic')

            # 2. solve the intermediate convex optimization problem
            kernel = get_kernel(psi, psi)
            svdd = SvddDualQP(kernel, self.C)
            svdd.train()
            threshold = svdd.get_threshold()
            inds = svdd.get_support_dual()
            alphas = svdd.get_support_dual_values()
            sol = psi[:,inds]*alphas

        self.sol = sol
        self.latent = latent
        return (sol, latent, threshold)

    def apply(self, pred_sobj):
        """ Application of the LatentSVDD:

            anomaly_score = min_z ||c*-\Psi(x,z)||^2
            latent_state = argmin_z ||c*-\Psi(x,z)||^2
        """

        N = pred_sobj.get_num_samples()
        norm2 = self.sol.trans()*self.sol

        vals = matrix(0.0, (1,N))
        lats = matrix(0.0, (1,N))
        for i in range(N):
            # min_z ||sol - Psi(x,z)||^2 = ||sol||^2 + min_z -2<sol,Psi(x,z)> + ||Psi(x,z)||^2
            # Hence => ||sol||^2 - max_z  2<sol,Psi(x,z)> - ||Psi(x,z)||^2
            (max_obj, lats[i], foo) = pred_sobj.argmax(self.sol, i, opt_type='quadratic')
            vals[i] = norm2 - max_obj

        return (vals, lats)
