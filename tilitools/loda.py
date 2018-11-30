import numpy as np
from pyod.models.base import BaseDetector


class LODA(BaseDetector):

    def __init__(self, contamination=0.1, n_bins=10, n_random_cuts=100, **kwargs):
        super(LODA, self).__init__(contamination=contamination)
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.weights = np.ones(n_random_cuts, dtype=np.float) / n_random_cuts

    def fit(self, X, y=None):
        n_components = X.shape[1]
        n_nonzero_components = np.sqrt(n_components)
        n_zero_components = n_components - np.int(n_nonzero_components)

        self.projections = np.random.randn(self.n_random_cuts, n_components)
        self.histograms = np.zeros((self.n_random_cuts, self.n_bins))
        self.limits = np.zeros((self.n_random_cuts, self.n_bins + 1))
        for i in range(self.n_random_cuts):
            rands = np.random.permutation(n_components)[:n_zero_components]
            self.projections[i, rands] = 0.
            projected_data = self.projections[i, :].dot(X.T)
            self.histograms[i, :], self.limits[i, :] = np.histogram(projected_data, bins=self.n_bins, density=False)
            self.histograms[i, :] += 1e-12
            self.histograms[i, :] /= np.sum(self.histograms[i, :])
        return self

    def decision_function(self, X):
        pred_scores = np.zeros([X.shape[0], 1])
        for i in range(self.n_random_cuts):
            projected_data = self.projections[i, :].dot(X.T)
            inds = np.searchsorted(self.limits[i, :self.n_bins - 1], projected_data, side='left')
            pred_scores[:, 0] += -self.weights[i] * np.log(self.histograms[i, inds])
        pred_scores /= self.n_random_cuts
        return pred_scores.ravel()