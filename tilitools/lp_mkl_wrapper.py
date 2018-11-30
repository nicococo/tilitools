import numpy as np

from tilitools.profiler import profile


class MKLWrapper:
    """Lp-norm Multiple Kernel Learning Wrapper for convex semi-supervised anomaly detection

        Note:
        - p-norm mkl supported
        - dual solution is supported.

        Written by: Nico Goernitz, TU Berlin, 2013/14
    """

    def __init__(self, ssad, kernels, pnorm=2.0):
        self.kernels = kernels  # (list of 2d arrays) kernel matrices
        self.samples = kernels[0].shape[0]
        self.pnorm = pnorm
        self.num_kernels = len(kernels)
        self.dm = np.ones(self.num_kernels, dtype=np.float) / np.float(self.num_kernels)  # (vector) mixing coefficients
        self.ssad = ssad
        self.ssad.set_train_kernel(self.combine_kernels(kernels))
        print('MKL with {0} kernels.'.format(self.num_kernels))

    def combine_kernels(self,kernels):
        dim1, dim2 = kernels[0].shape
        mixed = np.zeros((dim1, dim2))
        for i in range(self.num_kernels):
            mixed += self.dm[i] * kernels[i]
        return mixed

    @profile
    def fit(self, precision=1e-3):
        pnorm = self.pnorm
        iter = 0
        lastsol = np.zeros((self.num_kernels))
        while sum([abs(lastsol[i]-self.dm[i]) for i in range(self.num_kernels)]) > precision:
            # train ssad with current kernel mixing coefficients
            self.ssad.set_train_kernel(self.combine_kernels(self.kernels))
            self.ssad.fit()

            # calculate new kernel mixing coefficients
            lastsol = self.dm.copy()
            alphas = self.ssad.get_alphas()
            cy = self.ssad.cy

            # linear part of the objective
            norm_w_sq_m = np.zeros((self.num_kernels, 1))
            res = cy.dot(cy.T)*alphas.dot(alphas.T)
            for l in range(self.num_kernels):
                norm_w_sq_m[l] = np.sum(self.dm[l]*self.dm[l] * res * self.kernels[l])

            # solve the quadratic program
            sum_norm_w = np.sum(np.power(norm_w_sq_m, pnorm/(pnorm+1.0)))
            sum_norm_w = np.power(sum_norm_w, 1.0/pnorm)

            dm = np.power(norm_w_sq_m, 1.0/(pnorm+1.0))/sum_norm_w

            print('New mixing coefficients:', dm)
            dm_norm = np.sum(np.power(abs(dm), pnorm))
            dm_norm = np.power(dm_norm, 1.0/pnorm)

            print('Norm of mixing coefficients: ', dm_norm)
            self.dm = dm
            iter += 1
        print('Num iterations = {0}.'.format(iter))
        return self

    def get_threshold(self):
        return self.ssad.get_threshold()

    def get_support_dual(self):
        return self.ssad.get_support_dual()

    def get_mixing_coefficients(self):
        return self.dm

    def apply(self, kernels):
        mixed = self.combine_kernels(kernels)
        res = self.ssad.apply(mixed)
        return res
