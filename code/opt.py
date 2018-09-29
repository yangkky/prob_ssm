import itertools
from collections import Counter
import multiprocessing as mp
import operator

import numpy as np
import torch

import helpers
import objectives

def mod_lower(X, fn, perm, *args, **kwargs):
    """ Modular lower bound of fn(X) for any X contained in ground set V

    Expects X as a list of tuples, fn as a Python function,
    """
    low = 0.0
    for elem in X:
        i = perm.index(elem)
        low += fn(perm[:i + 1], *args, **kwargs)
        if i != 0:
            low -= fn(perm[:i], *args, **kwargs)
    return low

def mod_upper(X, fn, center, ground, *args, **kwargs):
    """ Modular upper bound of fn(X) for any X contained in ground set V,
    centered at center.

    Expects X and center as lists of tuples, and fn as a Python function. """

    f_center = fn(center, *args, **kwargs)
    f_ground = fn(ground, *args, **kwargs)
    f_empty = fn([], *args, **kwargs)
    up1 = fn(center, *args, **kwargs)
    up2 = fn(center, *args, **kwargs)

    for j in center:
        if j not in X:
            center_noj = [x for x in center if x != j]
            up1 -= f_center - fn(center_noj, *args, **kwargs)
            ground_noj = [x for x in ground if x != j]
            up2 -= f_ground - fn(ground_noj, *args, **kwargs)

    for j in X:
        if j not in center:
            up1 += fn([j], *args, **kwargs) - f_empty
            center_union_j = center + [j]
            up2 += fn(center_union_j, *args, **kwargs) - f_center
    return min((up1, up2))

def _make_best_permutation(V, X, f, f_args=[], f_kwargs={}):
    fx = f(X, *f_args, **f_kwargs).item()
    x_gains = [fx - f([x for x in X if x != j], *f_args, **f_kwargs).item()
               for j in X]
    V_not_X = [v for v in V if v not in X]
    v_gains = [f(X + [j], *f_args, **f_kwargs).item() for j in V_not_X]
    x_inds = np.argsort(x_gains)#[::-1]
    x_inds = [V.index(X[i]) for i in x_inds]
    v_inds = np.argsort(v_gains)[::-1]
    v_inds = [V.index(V_not_X[i]) for i in v_inds]
    return x_inds + v_inds



def _make_permuted_indices(V, X, f, f_args=[], f_kwargs={}):
    m = len(X)
    n = len(V)
    X_inds = [V.index(x) for x in X]
    not_X_inds = [i for i, v in enumerate(V) if v not in X]

    indices = np.array([_make_best_permutation(V, X, f, f_args=f_args, f_kwargs=f_kwargs)
                        for _ in range(max(n - m, m))])
    if m == 0 or m == n:
        return indices

    for i in range(max(m, n - m)):
        indices[i, m - 1], indices[i, i % m] = indices[i, i % m], indices[i, m - 1]
        # np.random.shuffle(indices[i, :m - 1])
        indices[i, m], indices[i, i % (n - m) + m] = indices[i, i % (n - m) + m], indices[i, m]
        # np.random.shuffle(indices[i, m + 1:])
    return indices

def _get_candidates(perm, V, X, uppers, obj,
                  g, g_args, g_kwargs):
    candidate = X[:]
    changed = False
    for i, upper in zip(V, uppers):
        if i in X:
            X_noi = [x for x in X if x != i]
            lower = mod_lower(X_noi, g, perm, *g_args, **g_kwargs)
            if upper - lower < obj:
                candidate.remove(i)
                changed = True
        else:
            lower = mod_lower(X + [i], g, perm, *g_args, **g_kwargs)
            if upper - lower < obj:
                candidate.append(i)
                changed = True
    return (candidate, changed)

def mod_mod(V, X0, fn, g, fn_args=[], g_args=[],
            fn_kwargs={}, g_kwargs={}, verbose=True):
    """ Implements algorithm3 (ModMod) from Ilyer and Bilmes (2013).

    Required arguments:
      V (list): the ground set as a list
      fn: submodular function; LHS of objective
      g: submodular function; RHS of objective
      X0 (list): subset of V to use as starting point for optimization

    Optional keyword arguments:
      fn_args (tuple): additional arguments to fn
      fn_kwargs (dict): keyword arguments to fn
      g_args (tuple): additional arguments to g
      g_kwargs (dict): keyword arguments to g
      verbose (Boolean): Whether to print objective every iteration. Default True.
    Returns:
      X_list (list): set at each iteration
      obj_list (list): objective at each iteration
    """

    X = X0
    obj_list = [] # stores objective at each time step
    X_list = [X]
    perm_list = []
    it = 0

    up = fn(X, *fn_args, **fn_kwargs)
    low = g(X, *g_args, **g_kwargs)
    obj = up - low
    obj_list.append(obj)

    if verbose:
        print('Iteration %d\t obj = %f\t' %(it, obj))

    while True:
        perms = _make_permuted_indices(V, X, objectives.objective,
                                       f_args=g_args[:-1], f_kwargs=g_kwargs)
        perms = [[V[i] for i in p] for p in perms]
        perm_list.append(perms)
        it += 1
        uppers = []
        for i in V:
            if i in X:
                X_noi = [x for x in X if x != i]
                uppers.append(mod_upper(X_noi, fn, X, V, *fn_args, **fn_kwargs))
            else:
                uppers.append(mod_upper(X + [i], fn, X, V, *fn_args, **fn_kwargs))

        with mp.Pool(6) as pool:
            candidates = [pool.apply_async(_get_candidates, args=(perm, V, X, uppers,
                                                                  obj_list[-1], g,
                                                                  g_args, g_kwargs))
                          for perm in perms]
            candidates = [c.get() for c in candidates]
        candidates = [c[0] for c in candidates if c[1]]
        if not candidates:
            break

        else:
            best = obj_list[-1]
            X_next = None
            for candidate in candidates:
                up = fn(candidate, *fn_args, **fn_kwargs)
                low = g(candidate, *g_args, **g_kwargs)
                obj = up - low
                if obj < best:
                    best = obj
                    X_next = candidate
            if verbose:
                print('Iteration %d\t obj = %f' %(it, best))
            obj_list.append(best)
            X = X_next
            X_list.append(X)

    return X_list, obj_list, perm_list, perms

def greedy(V, X0, objective, depth, obj_args=[], obj_kwargs={}, return_all=False):
    X = X0[:] # library X starts with seed
    Xs = [X[:]]
    obj = objective(X, *obj_args, **obj_kwargs)
    obj_list = [obj]
    doubles = [[_ for _ in itertools.product(V, repeat=d)]
               for d in range(1, depth + 1)]
    doubles = list(itertools.chain.from_iterable(doubles))
    while True:
        objs = [_get_deltas(objective, double, X, obj_args, obj_kwargs)
                for double in doubles]
        ind = np.argmin(objs)
        obj_next = min(objs)

        if obj_next >= obj:
            break
        else:
            for a in doubles[ind]:
                if a in X:
                    X.remove(a)
                else:
                    X.append(a)
            obj = obj_next
        Xs.append(X[:])
        obj_list.append(obj)
    if return_all:
        return X, obj, Xs, obj_list
    else:
        return X, obj

def _get_deltas(objective, aa, X, obj_args, obj_kwargs):
    A = X[:]
    for a in aa:
        if a in A:
            A.remove(a)
        else:
            A.append(a)
    return objective(A, *obj_args, **obj_kwargs)
