import time
import resource
import numpy as np

__version__ = '0.0.1'
__author__ = 'Nico Goernitz'
__date__ = '11.2017'

"""
    Mini runtime profiler using decorators.
"""

tilitools_profiles = None


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
    fname = fname[fname.rfind('/') + 1:-3]

    # dictionary key.
    # assumes that only one function with name 'name' exists
    # in file 'fname'
    fkey = '{0}'.format(fname)
    key = '{0}'.format(name)

    # from settings import my_list
    import tilitools.profiler as utils
    if utils.tilitools_profiles is None:
        utils.tilitools_profiles = {}

    if fkey in utils.tilitools_profiles:
        fcalls, ftime, fdict = utils.tilitools_profiles[fkey]
        if key not in fdict:
            fdict[key] = 0, 0., 0, 0
            utils.tilitools_profiles[fkey] = fcalls, ftime, fdict
    else:
        fdict = dict()
        fdict[key] = 0, 0., 0, 0
        utils.tilitools_profiles[fkey] = 0, 0., fdict

    def timed(*args, **kw):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t = time.time()
        result = fn(*args, **kw)
        t = time.time() - t
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem

        fcalls, ftime, fdict = utils.tilitools_profiles[fkey]
        ncalls, ntime, nmem, skip = fdict[key]
        if ncalls == 0:
            skip = t
        fdict[key] = ncalls + 1, ntime + t, max(nmem, mem), skip
        utils.tilitools_profiles[fkey] = fcalls + 1, ftime + t, fdict
        return result
    return timed


def print_profiles():
    """
    This function does provide a nice text-based summary of the
    global profile and should therefore be called only once, before the
    programs quits.
    """
    import tilitools.profiler as utils
    print(utils.tilitools_profiles)
    for fkey in utils.tilitools_profiles:
        fcalls, ftime, fdict = utils.tilitools_profiles[fkey]
        if fcalls == 0:
            print('\n-------{0}: unused.'.format(fkey.ljust(34)))
        else:
            print('\n-------{0}: ncalls={1:3d} total_time={2:1.4f} avg_time={3:1.4f}'.format(
                fkey.ljust(34), fcalls, ftime, ftime / float(fcalls)))

        keys = list(fdict.keys())
        times = list()
        for k in keys:
            ncalls, ntime, max_mem, skip = fdict[k]
            times.append(-ntime)

        sidx = np.argsort(times).tolist()
        for i in sidx:
            ncalls, ntime, max_mem, first_call = fdict[keys[i]]
            if ncalls == 0:
                print('      -{0}: unused.'.format(keys[i].ljust(34)))
            else:
                print('      -{0}: ncalls={1:3d} total_time={2:1.4f} first_call={3:1.4f} '
                      'avg_time={4:1.4f} max_mem={5}'.format(
                        keys[i].ljust(34), ncalls, ntime, first_call, ntime / float(ncalls), max_mem))
