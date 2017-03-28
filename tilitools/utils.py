__version__ = '0.0.1'
__author__ = 'Nico Goernitz'
__date__ = '12.2015'

import numpy as np
import numba
import time, resource

global_profiles = dict()  # contains the global profile information


@numba.autojit(nopython=True)
def argwhere_values_in_array(vals, arr):
    # 1. assume that vals are in arr
    # 2. assume that vals is exactly once in arr
    # 3. assume vals.size < arr.size
    inds = -np.ones(vals.size)
    for i in range(arr.size):
        for j in range(vals.size):
            if arr[i] == vals[j]:
                inds[j] == i
    return inds


def profile(fn=None):
    """
    This method is a decorator that keeps track of execution times
    and function calls of both, the function itself as well as the
    source code file (which also means that within each file only one
    method with the same name is allowed).
    Does not take care of subdirectories.
    Args:
        fn: decorated function
    Returns: wrapped timer function around 'func'
    """

    # name of the function
    name = fn.__name__

    # get the name of the file of the function
    fname = fn.__code__.co_filename
    fname = fname[fname.rfind('/')+1:-3]

    # dictionary key.
    # assumes that only one function with name 'name' exists
    # in file 'fname'
    fkey = '{0}'.format(fname)
    key = '{0}'.format(name)

    if global_profiles.has_key(fkey):
        if global_profiles.has_key(fkey):
            fcalls, ftime, fdict = global_profiles[fkey]
            if not fdict.has_key(key):
                fdict[key] = 0, 0., 0, 0
                global_profiles[fkey] = fcalls, ftime, fdict
    else:
        fdict = dict()
        fdict[key] = 0, 0., 0, 0
        global_profiles[fkey] = 0, 0., fdict

    def timed(*args, **kw):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t = time.time()
        result = fn(*args, **kw)
        t = time.time() - t
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem

        fcalls, ftime, fdict = global_profiles[fkey]
        ncalls, ntime, nmem, skip = fdict[key]
        if ncalls==0:
            skip = t
        fdict[key] = ncalls + 1, ntime + t, max(nmem, mem), skip
        global_profiles[fkey] = fcalls + 1, ftime + t, fdict
        return result
    return timed


def print_profiles():
    """
    This function does provide a nice text-based summary of the
    global profile and should therefore be called only once, before the
    programs quits.
    """
    for fkey in  global_profiles.keys():
        fcalls, ftime, fdict = global_profiles[fkey]
        if fcalls==0:
            print('\n-------{0}: unused.'.format(fkey.ljust(34)))
        else:
            print('\n-------{0}: ncalls={1:3d} total_time={2:1.4f} avg_time={3:1.4f}'.format( \
                fkey.ljust(34), fcalls, ftime, ftime/float(fcalls)))

        keys = fdict.keys()
        times = list()
        for i in range(len(keys)):
            ncalls, ntime, max_mem, skip = fdict[keys[i]]
            times.append(-ntime)

        sidx = np.argsort(times).tolist()
        for i in sidx:
            ncalls, ntime, max_mem, first_call = fdict[keys[i]]
            if ncalls==0:
                print('      -{0}: unused.'.format(keys[i].ljust(34)))
            else:
                print('      -{0}: ncalls={1:3d} total_time={2:1.4f} first_call={3:1.4f} avg_time={4:1.4f} max_mem={5}'.format( \
                    keys[i].ljust(34), ncalls, ntime, first_call, ntime/float(ncalls), max_mem))