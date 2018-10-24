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

def _make_swaps(perm, m):
    n = len(perm)
    X_swaps = itertools.combinations(range(m), 2)
    V_swaps = itertools.combinations(range(m, n), 2)
    for i, j in itertools.chain(X_swaps, V_swaps):
        yield perm[:i] + [perm[j]] + perm[i + 1:j] + [perm[i]] + perm[j + 1:]


def _minimize(perm, m, V, uppers, f, f_args, f_kwargs):
    best = torch.tensor(0.0)
    best_perm = perm
    X = perm[:m]
    best_X = X
    converged = False
    it = 1
    while not converged:
        print(it, best)
        it += 1
        converged = True
        perms = list(_make_swaps(best_perm, m))
        with mp.Pool(6) as pool:
            candidates = [pool.apply_async(_get_candidates, args=(perm, V, X, uppers,
                                                                  best, f,
                                                                  f_args, f_kwargs))
                          for perm in perms]
            candidates = [c.get() for c in candidates]
        perms = [p for p, c in zip(perms, candidates) if c[1]]
        candidates = [c[0] for c in candidates if c[1]]

        for candidate, perm in zip(candidates, perms):
            obj = f(candidate, *f_args)
            if obj < best:
                converged = False
                best = obj
                best_X = candidate
                best_perm = perm
    return best_perm, best_X, best



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

def mod_mod2(V, X0, objective, args=[], dec='dc', alpha=None, beta=None, verbose=True):
    X = X0
    obj_list = [] # stores objective at each time step
    X_list = [X]
    perm_list = []
    it = 0
    obj = objective(X, *args)
    obj_list.append(obj)

    if verbose:
        print('Iteration %d\t obj = %f\t' %(it, obj))

    left_kwargs = {
            'alpha':alpha,
            'beta':beta,
            'side':'left',
            'dec':dec
        }
    right_kwargs = {
            'alpha': alpha,
            'beta': beta,
            'side': 'right',
            'dec': dec
        }

    while True:
        # p = _make_best_permutation(V, X, objective, f_args=args)
        # p = [V[i] for i in p]
        X_ = X[:]
        V_ = [j for j in V if j not in X_]
        np.random.shuffle(X_)
        np.random.shuffle(V_)
        p = X_ + V_
        it += 1
        uppers = []
        for i in V:
            if i in X:
                X_noi = [x for x in X if x != i]
                uppers.append(mod_upper(X_noi, objective, X, V,
                                        *args, **left_kwargs))
            else:
                uppers.append(mod_upper(X + [i], objective, X, V,
                                        *args, **left_kwargs))
        perm, X_next, obj = _minimize(p, len(X), V, uppers,
                                      objective, args, right_kwargs)
        if obj >= obj_list[-1]:
            break
        else:
            obj_list.append(obj)
            X = X_next
            X_list.append(X)
            perm_list.append(perm)
            if verbose:
                print('Iteration %d\t obj = %f' %(it, obj))

    return X_list, obj_list, perm_list, perm

def mod_mod(V, X0, objective, args=[], dec='dc', alpha=None, beta=None, verbose=True):
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
    obj = objective(X, *args)
    obj_list.append(obj)

    if verbose:
        print('Iteration %d\t obj = %f\t' %(it, obj))

    left_kwargs = {
            'alpha':alpha,
            'beta':beta,
            'side':'left',
            'dec':dec
        }
    right_kwargs = {
            'alpha': alpha,
            'beta': beta,
            'side': 'right',
            'dec': dec
        }

    while True:
        perms = _make_permuted_indices(V, X, objective, f_args=args)
        perms = [[V[i] for i in p] for p in perms]
        perm_list.append(perms)
        it += 1
        uppers = []
        for i in V:
            if i in X:
                X_noi = [x for x in X if x != i]
                uppers.append(mod_upper(X_noi, objective, X, V,
                                        *args, **left_kwargs))
            else:
                uppers.append(mod_upper(X + [i], objective, X, V,
                                        *args, **left_kwargs))

        with mp.Pool(6) as pool:
            candidates = [pool.apply_async(_get_candidates, args=(perm, V, X, uppers,
                                                                  obj_list[-1], objective,
                                                                  args, right_kwargs))
                          for perm in perms]
            candidates = [c.get() for c in candidates]
        candidates = [c[0] for c in candidates if c[1]]
        if not candidates:
            break

        else:
            best = obj_list[-1]
            X_next = None
            for candidate in candidates:
                obj = objective(candidate, *args)
                if obj < best:
                    best = obj
                    X_next = candidate
            if not best < obj_list[-1]:
                break
            if verbose:
                print('Iteration %d\t obj = %f' %(it, best))
            obj_list.append(best)
            X = X_next
            X_list.append(X)

    return X_list, obj_list

def seeded_stochastic_usm(V, X0, objective, obj_args=[], obj_kwargs={}):
    X = X0[:]
    Y = [v for v in V if v not in X]
    obj_X = objective(X, *obj_args, **obj_kwargs)
    obj_Y = objective(Y, *obj_args, **obj_kwargs)
    np.random.shuffle(Y)
    for y in Y:
        gain_plus = objective(X + [y], *obj_args, **obj_kwargs) - obj_X
        gain_minus = objective([v for v in Y if v != y], *obj_args, **obj_kwargs) - obj_Y
        a = max(gain_plus, 0)
        b = max(gain_minus, 0)
        # print(y, a, b)
        if a == b == 0:
            a = 1
            b = 1
        p = a / (a + b)
        if np.random.random() < p:
            X = X + [y]
            X_obj = objective(X, *obj_args, **obj_kwargs)
        else:
            Y = [v for v in Y if v != y]
            obj_Y = objective(Y, *obj_args, **obj_kwargs)
    return Y, objective(Y, *obj_args, **obj_kwargs)

def supsub(V, X0, objective, args=[], dec='ds', alpha=None, beta=None, verbose=True):
    X = X0
    obj_list = [] # stores objective at each time step
    X_list = [X]
    it = 0
    obj = objective(X, *args)
    obj_list.append(obj)

    if verbose:
        print('Iteration %d\t obj = %f\t' %(it, obj))

    left_kwargs = {
            'alpha':alpha,
            'beta':beta,
            'side':'left',
            'dec':dec
        }
    right_kwargs = {
            'alpha': alpha,
            'beta': beta,
            'side': 'right',
            'dec': dec
        }
    while True:
        it += 1
        def anon(cand):
            g = objective(cand, *args, **right_kwargs)
            m = mod_upper(cand, objective, X, V, *args, **left_kwargs)
            return m - g

        def anon2(cand):
            return -anon(cand) + 100000
        X_new1, _ = seeded_stochastic_usm(V, [], anon2)
        obj1 = objective(X_new1, *args)
        X_new2, _ = greedy(V, X0, anon)
        X_new2 = X_new2[-1]
        obj2 = objective(X_new2, *args)
        # print(obj1, obj2)
        if obj1 < obj2:
            X_new = X_new1
            obj = obj1
        else:
            X_new = X_new2
            obj = obj2
        if set(X) == set(X_new):
            break
        else:
            X = X_new
            obj_list.append(obj)
            X_list.append(X)
            if verbose:
                print('Iteration %d\t obj = %f\t' %(it, obj))
    return X_list, obj_list

def greedy_add(V, X0, objective, obj_args=[], obj_kwargs={}, remove=False):
    X = X0[:]
    Xs = [X[:]]
    obj = objective(X, *obj_args, **obj_kwargs)
    obj_list = [obj]
    while True:
        if not remove:
            candidates = [X + [v] for v in V if v not in X]
        else:
            candidates = [[x for x in X if x != v] for v in X]
        if not candidates:
            return Xs, obj_list
        objs = [objective(c, *obj_args, **obj_kwargs) for c in candidates]
        ind = np.argmin(objs)
        obj_next = min(objs)
        if obj_next >= obj:
            break
        else:
            X = candidates[ind]
            obj = obj_next
            Xs.append(X)
            obj_list.append(obj)
    return Xs, obj_list



def greedy(V, X0, objective, depth=1, obj_args=[], obj_kwargs={}):
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
    return Xs, obj_list

def _get_deltas(objective, aa, X, obj_args, obj_kwargs):
    A = X[:]
    for a in aa:
        if a in A:
            A.remove(a)
        else:
            A.append(a)
    return objective(A, *obj_args, **obj_kwargs)
