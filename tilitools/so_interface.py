import numpy as np
from abc import ABCMeta, abstractmethod


class SOInterface:
    """ Structured Object Interface
    """
    __metaclass__ = ABCMeta  # This is an abstract base class (ABC)

    def __init__(self, X, y=None):
        self.X = X  # (list of np.arrays) data
        self.y = y  # (list of np.arrays) state sequences (if present)

        # assume either numpy array or list-of-objects
        if isinstance(X, list):
            self.samples = len(X)
            self.feats = X[0].shape[0]
        else:
            self.feats, self.samples = X.shape
        print('Structured object with #{0} samples and #{1} features.'.format(self.samples, self.feats))

    def get_hotstart_sol(self):
        print('Generate a random solution vector for hot start.')
        return np.random.randn(self.get_num_dims(), 1)

    def get_num_samples(self):
        return self.samples

    @abstractmethod
    def get_num_states(self):
        raise NotImplementedError

    @abstractmethod
    def get_num_dims(self):
        raise NotImplementedError

    @abstractmethod
    def argmax(self, sol, idx, add_loss=False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self, idx, y):
        raise NotImplementedError

    @abstractmethod
    def get_joint_feature_map(self, idx, y=None):
        raise NotImplementedError

