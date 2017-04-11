import numpy as np

from numba import autojit


@autojit(nopython=False)
def optimize_subgradient_descent(x0, fun, fgrad, max_iter, prec, rate):
    """ Subgradient descent solver. Optimized for numba.

        Solves:
                min_x f(x)  for non-smooth f
    """
    dims = x0.size
    x = x0
    best_x = x
    best_obj = np.float64(1e20)

    obj_bak = -100.
    iter = 0
    is_converged = False
    while not is_converged and iter < max_iter:
        obj = fun(x)
        # this is subgradient, hence need to store the best solution so far
        if best_obj >= obj:
            best_x = x
            best_obj = obj

        # stop, if progress is too slow
        if np.abs((obj-obj_bak)/obj) < prec:
            is_converged = True
            continue
        obj_bak = obj

        # gradient step for threshold
        grad = fgrad(x)
        norm_grad = 0.0
        for d in range(dims):
            norm_grad += grad[d]*grad[d]
        norm_grad = np.sqrt(norm_grad)

        # dimishing stepsize
        max_change = rate / np.float(iter+1)*10.

        for d in range(dims):
            x[d] -= grad[d] * max_change / norm_grad
        iter += 1
    return best_x, best_obj, iter