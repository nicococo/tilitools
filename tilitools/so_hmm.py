import numpy as np

from tilitools.so_interface import SOInterface
from tilitools.profiler import profile


class SOHMM(SOInterface):
    """ Hidden Markov Structured Object."""
    ninf = -10.0**15
    start_p = None  # (vector) start probabilities
    states = -1  # (scalar) number transition states
    hotstart_tradeoff = 0.1  # (scalar) this tradeoff is used for hotstart
                             # > 1.0: transition have more weight
                             # < 1.0: emission have more weight

    def __init__(self, X, y=None, num_states=2, hotstart_tradeoff=0.1):
        SOInterface.__init__(self, X, y)
        self.states = num_states
        self.start_p = np.ones(self.states)
        self.hotstart_tradeoff = hotstart_tradeoff

    def get_hotstart_sol(self):
        sol = np.random.randn(self.get_num_dims())
        sol[:self.states*self.states] = self.hotstart_tradeoff
        print('Hotstart position uniformly random with '
              'transition tradeoff {0} and size {1}.'.format(self.hotstart_tradeoff, self.get_num_dims()))
        return sol

    @profile
    def calc_emission_matrix(self, sol, idx, augment_loss=False, augment_prior=False):
        T = self.X[idx][0, :].size
        N = self.states
        F = self.feats

        em = np.zeros((N, T))
        for t in range(T):
            for s in range(N):
                for f in range(F):
                    em[s,t] += sol[N*N + s*F + f] * self.X[idx][f, t]
        # augment with loss
        if augment_loss:
            loss = np.ones((N, T))
            for t in range(T):
                loss[self.y[idx][t], t] = 0.0
            em += loss
        return em

    @profile
    def get_transition_matrix(self, sol):
        N = self.states
        # transition matrix
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                A[i, j] = sol[i*N+j]
        return A

    @profile
    def argmax(self, sol, idx, add_loss=False, opt_type='linear'):
        # if labels are present, then argmax will solve
        # the loss augmented programm
        T = self.X[idx][0, :].size
        N = self.states
        F = self.feats

        # get transition matrix from current solution
        A = self.get_transition_matrix(sol)
        # calc emission matrix from current solution, data points and
        # augment with loss if requested
        em = self.calc_emission_matrix(sol, idx, augment_loss=add_loss)

        delta = np.zeros((N, T))
        psi = np.zeros((N, T))
        # initialization
        for i in range(N):
            delta[i, 0] = self.start_p[i] + em[i, 0]

        # recursion
        for t in range(1, T):
            for i in range(N):
                delta[i, t], psi[i, t] = max([(delta[j,t-1] + A[j,i] + em[i,t], j) for j in range(N)])

        states = np.zeros(T, dtype=np.int)
        prob, states[T-1]  = max([delta[i, T-1], i] for i in range(N))

        for t in reversed(range(1,T)):
            states[t-1] = psi[states[t], t]

        psi_idx = self.get_joint_feature_map(idx, states)
        val = sol.T.dot(psi_idx)
        return val, states, psi_idx

    @profile
    def get_jfm_norm2(self, idx, y=None):
        if y is None:
            y = self.y[idx]
        jfm = self.get_joint_feature_map(idx, y)
        return jfm.T.dot(jfm)

    def calc_loss(self, idx, y):
        return np.sum(self.y[idx] != y)

    @profile
    def get_scores(self, sol, idx, y=None):
        if y is None:
            y = self.y[idx]

        _, T = y.shape
        N = self.states
        F = self.feats
        scores = np.zeros((1, T))

        # this is the score of the complete example
        anom_score = sol.T.dot(self.get_joint_feature_map(idx))

        # transition matrix
        A = self.get_transition_matrix(sol)
        # emission matrix without loss
        em = self.calc_emission_matrix(sol, idx, augment_loss=False)

        # store scores for each position of the sequence
        scores[0] = self.start_p[int(y[0,0])] + em[int(y[0,0]),0]
        for t in range(1,T):
            scores[t] = A[int(y[0,t-1]),int(y[0,t])] + em[int(y[0,t]),t]

        # transform for better interpretability
        if np.max(np.abs(scores)) > 10.0**(-15):
            scores = np.exp(-np.abs(4.0*scores/np.max(np.abs(scores))))
        else:
            scores = np.zeros((1,T))

        return np.float(anom_score), scores

    @profile
    def get_joint_feature_map(self, idx, y=None):
        if y is None:
            y = self.y[idx]

        T = y.size
        N = self.states
        F = self.feats
        jfm = np.zeros(self.get_num_dims())

        # transition part
        for i in range(N):
            inds = np.where(y[1:T] == i)[0]
            for j in range(N):
                indsj = np.where(y[inds] == j)[0]
                jfm[j*N+i] = indsj.size

        # emission parts
        for t in range(T):
            for f in range(F):
                jfm[y[t]*F + f + N*N] += self.X[idx][f, t]
        return jfm

    def get_num_dims(self):
        return self.feats*self.states + self.states*self.states
