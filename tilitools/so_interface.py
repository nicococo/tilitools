import numpy as np
from abc import ABCMeta, abstractmethod

class SOInterface:
    """ Structured Object Interface
    """
    __metaclass__ = ABCMeta  # This is an abstract base class (ABC)

    X = None  # (list of np.arrays) data
    y = None  # (list of np.arrays) state sequences (if present)

    samples = -1  # (scalar) number of training data samples
    dims = -1     # (scalar) overall number of dimensions of the model (=! number of features)
    feats = -1    # (scalar) number of features for each sample

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

        # assume either numpy array or list-of-objects
        if isinstance(X, list):
            self.samples = len(X)
            self.feats, _ = X[0].shape
        else:
            self.feats, self.samples = X.shape
        print('Create structured object with #{0} training examples, each consiting of #{1} features.'.format(self.samples, self.feats))

    def get_hotstart_sol(self):
        print('Generate a random solution vector for hot start.')
        return np.random.randn(self.get_num_dims(), 1)

    def get_num_samples(self):
        return self.samples

    @abstractmethod
    def get_num_dims(self):
        raise NotImplementedError

    @abstractmethod
    def argmax(self, sol, idx, add_loss=False, opt_type='linear'):
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self, idx, y):
        raise NotImplementedError

    @abstractmethod
    def get_joint_feature_map(self, idx, y=None):
        raise NotImplementedError

