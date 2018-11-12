import numpy as np


def get_kernel(X, Y, type='linear', param=1.0):
    """Calculates a kernel given the data X and Y (dims x exms)"""
    Xdims, Xn = X.shape
    Ydims, Yn = Y.shape

    kernel = 1.0
    if type == 'linear':
        print('Calculating linear kernel with size {0}x{1}.'.format(Xn, Yn))
        kernel = X.T.dot(Y)

    if type == 'rbf':
        print('Calculating Gaussian kernel with size {0}x{1} and sigma2={2}.'.format(Xn, Yn, param))
        Dx = (np.ones((Yn, 1)) * np.diag(X.T.dot(X)).reshape(1, Xn)).T
        Dy = (np.ones((Xn, 1)) * np.diag(Y.T.dot(Y)).reshape(1, Yn))
        kernel = Dx - 2. * np.array(X.T.dot(Y)) + Dy
        kernel = np.exp(-kernel / param)

    return kernel


def get_diag_kernel(X, type='linear', param=1.0):
    """Calculates the kernel diagonal given the data X (dims x exms)"""
    (Xdims, Xn) = X.shape

    kernel = 1.0
    if type == 'linear':
        print('Calculating diagonal of a linear kernel with size {0}x{1}.'.format(Xn, Xn))
        kernel = np.sum(X*X, axis=0)

    if type == 'rbf':
        print('Gaussian kernel diagonal is always exp(0)=1.')
        kernel = np.ones(Xn, dtype='d')
    return kernel


def normalize_kernel(K):
    # A kernel K is normalized, iff K_ii = 1 \forall i
    N = K.shape[0]
    a = np.sqrt(np.diag(K)).reshape((N, 1))
    if any(np.isnan(a)) or any(np.isinf(a)) or any(np.abs(a) <= 1e-16):
        print('Numerical instabilities.')
        C = np.eye(N)
    else:
        b = 1. / a
        C = b.dot(b.T)
    return K * C


def center_kernel(K):
    # Mean free in feature space
    N = K.shape[0]
    a = np.ones((N, N)) / np.float(N)
    return K - a.dot(K) - K.dot(a) + a.dot(K.dot(a))


def kta_align_general(K1, K2):
    # Computes the (empirical) alignment of two kernels K1 and K2
    # Definition 1: (Empirical) Alignment
    #   a = <K1, K2>_Frob
    #   b = sqrt( <K1, K1> <K2, K2>)
    #   kta = a / b
    # with <A, B>_Frob = sum_ij A_ij B_ij = tr(AB')
    return K1.dot(K2.T).trace() / np.sqrt(K1.dot(K1.T).trace() * K2.dot(K2.T).trace())


def kta_align_binary(K, y):
    # Computes the (empirical) alignment of kernel K1 and
    # a corresponding binary label  vector y \in \{+1, -1\}^m
    m = np.int(y.size)
    YY = y.reshape((m, 1)).dot(y.reshape((1, m)))
    return K.dot(YY).trace() / (m * np.sqrt(K.dot(K.T).trace()))
