import numpy as np

from numba import jit

from tilitools.so_interface import SOInterface
from tilitools.profiler import profile


class SOHMM(SOInterface):
    """ Hidden Markov Structured Object.
    """
    ninf = 1e-15
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
    @jit(nopython=False)
    def argmax(self, sol, idx, add_loss=False, **kwargs):
        train_y = np.empty((0), dtype=np.int)
        if add_loss:
            train_y = self.y[idx]
        states = _viterbi_argmax(sol.ravel(), self.X[idx], train_y, self.states)
        psi_idx = _get_joint_feature_map(self.X[idx], states, self.states)
        val = sol.T.dot(psi_idx)
        return val, states, psi_idx

    def calc_loss(self, idx, y):
        return np.sum(self.y[idx] != y)

    @profile
    def get_joint_feature_map(self, idx, y=None):
        if y is None:
            y = self.y[idx]
        return _get_joint_feature_map(self.X[idx], y)

    def get_num_dims(self):
        return self.feats*self.states + self.states*self.states

    def get_num_states(self):
        return self.states


@profile
@jit(nopython=True)
def _viterbi_argmax(sol, X, y, states):
    # if labels are present, then argmax will solve
    # the loss augmented program
    T = X.shape[1]

    # get transition matrix from current solution
    A = np.zeros((states, states), dtype=np.double)
    for i in range(states):
        for j in range(states):
            A[i, j] = sol[i*states+j]

    # calc emission matrix from current solution, data points and
    F = X.shape[0]
    em = np.zeros((states, T))
    for t in range(T):
        for s in range(states):
            for f in range(F):
                em[s, t] += sol[states*states + s*F + f] * X[f, t]
    # augment with loss
    if y.size > 0:
        loss = np.ones((states, T))
        for t in range(T):
            loss[y[t], t] = 0.0
        em += loss

    delta = np.zeros((states, T))
    psi = np.zeros((states, T), dtype=np.int8)
    # initialization
    for i in range(states):
        # use equal start probs for each state
        delta[i, 0] = 0. + em[i, 0]

    # recursion
    for t in range(1, T):
        for i in range(states):
            foo_argmax = 0
            foo_max = -1e16
            for l in range(states):
                foo = delta[l, t-1] + A[l, i] + em[i, t]
                if foo > foo_max:
                    foo_max = foo
                    foo_argmax = l
            psi[i, t] = foo_argmax
            delta[i, t] = foo_max

    states = np.zeros(T, dtype=np.int8)
    states[T-1] = np.argmax(delta[:, T-1])

    # for t in reversed(xrange(1, T)):
    for t in range(T-1, 0, -1):
        states[t-1] = psi[states[t], t]
    return states


@profile
@jit(nopython=True)
def _get_joint_feature_map(X, y, states):
    T = y.size
    F = X.shape[0]
    jfm = np.zeros(states*states + states*F)
    # transition part
    for t in range(T-1):
        for i in range(states):
            for j in range(states):
                if y[t] == i and y[t+1] == j:
                    jfm[j*states+i] += 1
    # emission parts
    for t in range(T):
        for f in range(F):
            jfm[y[t]*F + f + states*states] += X[f, t]
    return jfm
