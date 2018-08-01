import itertools
from collections import Counter

import numpy as np
import torch

def mod_lower(X, fn, perm, *args, **kwargs):
    """ Modular lower bound of fn(X) for any X contained in ground set V
    with permutation chain perm (aka S).

    Expects X as a list of tuples, fn as a Python function,
    and perm as a list.
    """
    low = 0.0 # lower modular bound

    for elem in X:
        i = perm.index(elem)
        low += fn(perm[:i + 1], *args, **kwargs)
        if i != 0:
            low -= fn(perm[:i], *args, **kwargs)
    return low

def mod_upper(X, fn, center, *args, **kwargs):
    """ Modular upper bound of fn(X) for any X contained in ground set V,
    centered at center.

    Expects X and center as lists of tuples, and fn as a Python function. """

    up = fn(center, *args, **kwargs) # modular upper bound

    for j in center:
        if j not in X:
            center_noj = [x for x in center if x != j]
            up -= fn(center, *args, **kwargs) - fn(center_noj, *args, **kwargs)

    for j in X:
        if j not in center:
            up += fn([j], *args, **kwargs) - fn([], *args, **kwargs)

    return up

def make_perm(V, X, seed=None):
    """ Takes in the ground set V and a set X, and
    returns a random chain permutation that contains X """
    if seed is not None:
        np.random.seed(seed)
    if len(X) == 0:
        indices = list(range(len(V)))
        np.random.shuffle(indices)
        return [V[i] for i in indices]

    ind_X = [i for i, v in enumerate(V) if v in X] # indices of X in V
    rest = [i for i in list(range(len(V))) if i not in ind_X] # rest of indices in V
    np.random.shuffle(ind_X) # shuffle indices
    np.random.shuffle(rest)
    indices = ind_X + rest # combine

    return [V[i] for i in indices] # generate perm based on shuffled indices

def mod_mod(V, fn, g, seed, fn_args=None, g_args=None, fn_kwargs={}, g_kwargs={}):
    """ Implements algorithm3 (ModMod) from paper. Takes in ground set V,
    and functions fn and g.

    Expects V as a list of tuples, and fn and g as submodular Python functions. """

    X = seed
    obj_lst = [] # stores objectives at each time step
    it = 0
    perm = make_perm(V, X) # choose permutation
    up = mod_upper(X, fn, X, *fn_args, **fn_kwargs)
    low = mod_lower(X, g, perm, *g_args, **g_kwargs)
    emp = up - low
    obj_lst.append(emp)
    print('Iteration %d\t obj = %f\t' %(it, emp))

    while True:
        it += 1
        X_next = X[:]

        for i in V:
            if i in X:
                X_noi = [x for x in X if x != i]
                obj = mod_upper(X_noi, fn, X, *fn_args, **fn_kwargs)
                obj -= mod_lower(X_noi, g, perm, *g_args, **g_kwargs)
                if obj < emp:
                    X_next.remove(i)
            else:
                obj = mod_upper(X + [i], fn, X, *fn_args, **fn_kwargs)
                obj -= mod_lower(X + [i], g, perm, *g_args, **g_kwargs)
                if obj < emp:
                    X_next.append(i)
        perm = make_perm(V, X_next) # choose permutation
        up = mod_upper(X_next, fn, X_next, *fn_args, **fn_kwargs)
        low = mod_lower(X_next, g, perm, *g_args, **g_kwargs)
        emp = up - low
        obj_lst.append(emp)
        N = get_N(X_next, 4)
        print('Iteration %d\t obj = %f' %(it, emp))
        if obj_lst[-1] == obj_lst[-2]:
            break
        else:
            X = X_next

    return X_next, obj_lst

def get_N(X, L):
    N = 1 # represents the product of sequence of # aas at each position
    counts = Counter([x[1] for x in X])
    for i in range(L):
        N *= counts[i]
    return N


def obj_LHS(X, L, probs):
    """ Takes in library X, and probabilities.

    Expects X to be a list of tuples, and probs to be a dictionary.

    Returns LHS of objective to be maximized (a supermodular function):
    sum of probabilities. """

    N = get_N(X, L)
    if N == 0:
        return torch.tensor(0.0)

    # filter thru probs to find prob of x's in X
    X.sort(key=lambda tup: tup[1])

    X_str = [[tup[0] for i, tup in enumerate(X) if tup[1] == j] for j in range(4)] # generate list of lists of strings
    X_str = [''.join(s) for s in itertools.product(*X_str)] # generate list of strings of 4 aa seqs

    p = torch.Tensor([probs[key] for key in X_str])

    return -1 * torch.sum(p)

def obj_RHS(X, L, probs, n):
    """ Takes in library X, probabilities, and batch size n.

    Expects X to be a list of tuples, and probs to be a dictionary.

    Returns RHS of objective to be maximized (a submodular function):
    sum of probabilities times expression with N and n. """

    N = get_N(X, L)
    if N == 0:
        return torch.tensor(0.0)

    # filter thru probs to find prob of x's in X
    X.sort(key=lambda tup: tup[1])

    X_str = [[tup[0] for i, tup in enumerate(X) if tup[1] == j] for j in range(4)] # generate list of lists of strings
    X_str = [''.join(s) for s in itertools.product(*X_str)] # generate list of strings of 4 aa seqs

    p = torch.Tensor([probs[key] for key in X_str])
    obj = torch.sum(p) * (1 - 1 / N) ** n

    return -1 * obj
