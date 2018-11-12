import numpy as np

from numba import jit


@jit(nopython=False)
def min_subgradient_descent(x0, fun, fgrad, max_iter, prec, rate, step_method):
    """ Subgradient descent solver. Optimized for numba.
        Solves:
                min_x f(x)  for non-smooth f
    """
    dims = x0.size
    x = x0
    best_x = x
    best_obj = np.float64(1e20)
    obj_bak = -1e10
    iter = 0
    is_converged = False
    while not is_converged and iter < max_iter:
        obj = fun(x)
        # this is subgradient, hence need to store the best solution so far
        if best_obj >= obj:
            best_x = x
            best_obj = obj

        # stop, if progress is too slow
        if np.abs((obj-obj_bak)) < prec:
            is_converged = True
            continue
        obj_bak = obj

        # gradient step for threshold
        grad = fgrad(x)
        if step_method == 1:
            # constant step
            max_change = rate
        elif step_method == 2:
            # dimishing step size
            max_change = rate / np.float(iter+1)
        else:
            # const. step length
            max_change = rate / np.linalg.norm(grad)
        for d in range(dims):
            x[d] -= grad[d]*max_change
        iter += 1
    return best_x, best_obj, iter