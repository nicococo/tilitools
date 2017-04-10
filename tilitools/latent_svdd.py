import numpy as np

from svdd_dual_qp import SvddDualQP

class LatentSVDD:
    """ Latent variable support vector data description.
        Written by Nico Goernitz, TU Berlin, 2014

        For more information see:
        'Learning and Evaluation with non-i.i.d Label Noise'
        Goernitz et al., AISTATS & JMLR W&CP, 2014
    """
    PRECISION = 1e-3  # important: effects the threshold, support vectors and speed!

    nu = 1.0	 # (scalar) the regularization constant > 0
    sobj = None  # structured object contains various functions
              # i.e. get_num_dims(), get_num_samples(), get_sample(i), argmin(sol,i)
    sol = None  # (vector) solution vector (after training, of course)

    def __init__(self, sobj, nu=1.0):
        self.nu = nu
        self.sobj = sobj

    def fit(self, max_iter=50):
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

        sol = np.random.randn(DIMS).reshape((DIMS, 1))
        psi = np.zeros((DIMS, N))  # (dim x exm)
        old_psi = np.zeros((DIMS, N))  # (dim x exm)
        threshold = 0.
        obj = -1.
        iter = 0
        # terminate if objective function value doesn't change much
        while iter<max_iter and (iter<2 or sum(sum(abs(np.array(psi-old_psi))))>=0.001):
            print('Starting iteration {0}.'.format(iter))
            print(sum(sum(abs(np.array(psi-old_psi)))))
            iter += 1
            old_psi = psi

            # 1. linearize
            # for the current solution compute the
            # most likely latent variable configuration
            for i in range(N):
                # min_z ||sol - Psi(x,z)||^2 = ||sol||^2 + min_z -2<sol,Psi(x,z)> + ||Psi(x,z)||^2
                # Hence => ||sol||^2 - max_z  2<sol,Psi(x,z)> - ||Psi(x,z)||^2
                _, latent[i], foo = self.sobj.argmax(sol, i, opt_type='quadratic')
                psi[:, i] = foo.reshape((DIMS))

            # 2. solve the intermediate convex optimization problem
            svdd = SvddDualQP('linear', None, self.nu)
            svdd.fit(psi)
            threshold = svdd.get_radius()
            sol = psi.dot(svdd.alphas)
        self.sol = sol
        self.latent = latent
        return sol, latent, threshold

    def apply(self, pred_sobj):
        """ Application of the LatentSVDD:

            anomaly_score = min_z ||c*-\Psi(x,z)||^2
            latent_state = argmin_z ||c*-\Psi(x,z)||^2
        """
        N = pred_sobj.get_num_samples()
        norm2 = self.sol.T.dot(self.sol)

        vals = np.zeros((N))
        lats = [0]*N
        for i in range(N):
            # min_z ||sol - Psi(x,z)||^2 = ||sol||^2 + min_z -2<sol,Psi(x,z)> + ||Psi(x,z)||^2
            # Hence => ||sol||^2 - max_z  2<sol,Psi(x,z)> - ||Psi(x,z)||^2
            max_obj, lats[i], foo = pred_sobj.argmax(self.sol, i, opt_type='quadratic')
            vals[i] = norm2 - max_obj

        return vals, lats
