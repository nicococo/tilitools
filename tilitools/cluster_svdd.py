__author__ = 'nicococo'
import numpy as np


class ClusterSvdd:
    """ Implementation of the cluster support vector data description (ClusterSVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """

    def __init__(self, svdds, nu=-1.0):
        self.clusters = len(svdds)
        self.svdds = svdds
        self.nu = nu
        self.use_local_fraction = nu <= 0.
        print('Creating new ClusterSVDD with {0} clusters.'.format(self.clusters))

    def fit(self, X, min_chg=0.0, max_iter=40, max_svdd_iter=2000, init_membership=None):
        """
        :param X: Data matrix is assumed to be feats x samples.
        :param min_chg: Minimum percent of changes per iteration before stopping.
        :param max_iter: Maximum number of iteration before stopping.
        :param max_svdd_iter: Maximum number of iterations for nested SVDDs.
        :param init_membership: Integer array with cluster affiliation per
                                sample (used for initialization).
        :return: (Integer array ) Cluster affiliations for all samples.
        """
        dims, samples = X.shape

        # init majorization step
        cinds_old = np.zeros(samples)
        cinds = np.random.randint(0, self.clusters, samples)
        if init_membership is not None:
            print('Using init cluster membership.')
            cinds = init_membership

        # init maximization step
        for c in range(self.clusters):
            inds = np.where(cinds == c)[0]
            self.svdds[c].fit(X[:, inds])

        iter_cnt = 0
        scores = np.zeros((self.clusters, samples))
        while np.sum(np.abs(cinds_old-cinds))/np.float(samples) > min_chg and iter_cnt < max_iter:
            print('Iter={0}'.format(iter_cnt))
            # 1. majorization step
            for c in range(self.clusters):
                scores[c, :] = self.svdds[c].predict(X)
            cinds_old = cinds
            cinds = np.argmin(scores, axis=0)
            # 2. maximization step
            for c in range(self.clusters):
                inds = np.where(cinds == c)[0]
                if inds.size > 0:
                    # perc = 2.0*np.float(inds.size)/np.float(samples)
                    # self.svdds[c].nu = perc * self.nu
                    self.svdds[c].fit(X[:, inds], max_iter=max_svdd_iter)
            iter_cnt += 1
        print('ClusterSVDD training finished after {0} iterations.'.format(iter_cnt))
        return cinds

    def predict(self, Y):
        """
        :param Y:
        :return:
        """
        scores = np.zeros((self.clusters, Y.shape[1]))
        for c in range(self.clusters):
            scores[c, :] = self.svdds[c].predict(Y)
        cinds = np.argmin(scores, axis=0)
        return np.min(scores, axis=0), cinds
