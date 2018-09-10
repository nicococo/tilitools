import numpy as np

from utils import profile

class MKLWrapper:
    """Lp-norm Multiple Kernel Learning Wrapper for convex semi-supervised anomaly detection

        Note:
        - p-norm mkl supported
        - dual solution is supported.

        Written by: Nico Goernitz, TU Berlin, 2013/14
    """
    samples = -1  # (scalar) number of samples
    pnorm = 2.0     # (scalar) mixing coefficient regularizer norm
    kernels = None  # (3-tensor) (=list of cvxopt.matrix) kernel matrices
    dm = None  # (vector) kernel mixing coefficients
    ssad = None  # (method)
    num_kernels = 0  # (scalar) number of kernels used

    def __init__(self, ssad, kernels, pnorm=2.0):
        self.kernels = kernels
        self.samples = kernels[0].shape[0]
        self.pnorm = pnorm
        self.num_kernels = len(kernels)
        self.dm = np.ones((self.num_kernels), dtype=np.float) / np.float(self.num_kernels)
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
            #
            # for j in range(self.samples):
            #     for k in range(self.samples):
            #         foo = float(cy[k])*float(cy[j])*alphas[k]*alphas[j]
            #         for l in range(self.num_kernels):
            #             norm_w_sq_m[l] += self.dm[l]*self.dm[l]*foo*self.kernels[l][j,k]

            # solve the quadratic programm
            sum_norm_w = np.sum(np.power(norm_w_sq_m, pnorm/(pnorm+1.0)))
            # for i in range(self.num_kernels):
            #     sum_norm_w += np.power(norm_w_sq_m[i], pnorm/(pnorm+1.0))
            sum_norm_w = np.power(sum_norm_w, 1.0/pnorm)

            dm = np.power(norm_w_sq_m, 1.0/(pnorm+1.0))/sum_norm_w
            # for i in range(self.num_kernels):
            #     dm[i] = np.power(norm_w_sq_m[i], 1.0/(pnorm+1.0))/sum_norm_w

            print('New mixing coefficients:')
            print(dm)

            dm_norm = np.sum(np.power(abs(dm), pnorm))
            # for i in range(self.num_kernels):
            #     dm_norm += np.power(abs(dm[i]), pnorm)
            dm_norm = np.power(dm_norm, 1.0/pnorm)

            print(dm_norm)
            self.dm = dm
            iter+=1

        print('Num iterations = {0}.'.format(iter))
        return 0

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
