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
        kernel = Dx - 2.* np.array(X.T.dot(Y)) + Dy
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


def center_kernel(K):
    print('IMPLEMENTED ME')
    return K


def normalize_kernel(K):
    print('IMPLEMENTED ME')
    return K