import numpy as np

from tilitools.so_interface import SOInterface
from tilitools.profiler import profile


class SOMultiClass(SOInterface):
    """ Multi class structured object."""
    num_classes = -1  # (scalar) number of classes

    def __init__(self, X, classes, y=None):
        # the class also acts as indices therefore: y >= 0!
        SOInterface.__init__(self, X, y)
        self.num_classes = classes

    @profile
    def argmax(self, sol, idx, add_loss=False, opt_type='linear'):
        nd = self.feats
        d = 0  # start of dimension in sol
        val = -10.**10.
        cls = -1 # best class

        for c in range(self.num_classes):
            foo = sol[d:d+nd].T.dot(self.X[:, idx])
            # the argmax of the above function
            # is equal to the argmax of the quadratic function
            # foo = + 2*foo - normPsi
            # since ||\Psi(x_i,z)|| = ||\phi(x_i)|| = y \forall z
            d += nd
            if np.single(foo) > np.single(val):
                val = foo
                cls = c
        if opt_type == 'quadratic':
            normPsi = self.X[:, idx].T.dot(self.X[:, idx])
            val = 2.*val - normPsi

        psi_idx = self.get_joint_feature_map(idx, cls)
        return val, cls, psi_idx

    @profile
    def calc_loss(self, idx, y):
        if self.y[idx] == y:
            return 0.
        return 1.

    @profile
    def get_joint_feature_map(self, idx, y=-1):
        if y == -1:
            y = self.y[idx]
        nd = self.feats
        mc = self.num_classes
        psi = np.zeros((nd*mc))
        psi[nd*y:nd*(y+1)] = self.X[:, idx]
        return psi

    def get_num_dims(self):
        return self.feats*self.num_classes