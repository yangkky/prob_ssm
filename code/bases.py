import torch
from torch import distributions as dist
import itertools

import operator

import numpy as np

from gptorch import kernels, models
import helpers, opt


# USES/RETURNS STRINGS, RETURNS ALL 24 POSSIBILITIES
def det_fixed(wt, X, y): # deterministic
    """ Takes in wildtype sequence, X, and y to compute baseline that creates
    optimal sequence from X's given optimal amino acids (those with max
    y-values) at each position out of the four possible positions in the
    wildtype sequence by fixing the three other positions, then continues onto
    the next position in the wildtype sequence by fixing the best amino acid in
    the previous position. So the fixed substring is not necessarily a fixed
    substring of the wildtype sequence.

    Note: wildtype sequence expected as string. X expected as an array or list
    of one-hot encodings.

    Returns list of all possible 24 optimal untested variants (as a string). """

    X_decode = [helpers.decode_X(x) for x in X]
    baseline = []

    # list of all possible permutations for picking which aa to vary
    perms = list(itertools.permutations(np.arange(len(wt))))
    for perm in perms:
        # seq starts out as wt seq, will store variant after iteration thru perm
        seq = list(wt)

        for i in perm:
            fixed = ''.join(seq) # fixed substring

            # index of xs in X with fixed substring
            index = [j for j, x in enumerate(X_decode) if fixed[0:i] == x[0:i] \
                and fixed[i + 1:len(fixed)] == x[i + 1:len(x)]]
            # stores y values of x's in X with those 3 fixed amino acids
            ys = [y[j] for j in index]

            # takes first occurrence of index with maximum y value
            max_ind = np.where(ys==max(ys))[0][0]

            seq[i] = X_decode[index[max_ind]][i]

        baseline.append(''.join(seq))

    return baseline


# USES/RETURNS STRINGS
def det_vary(wt, X, y): # deterministic
    """ Takes in wildtype sequence, X, and y to compute baseline that creates
    optimal sequence from X's given optimal amino acids (those with max
    y-values) at each position out of the four possible positions in the
    wildtype sequence by fixing the three other positions, then takes the best
    amino acid at each position. The fixed substring in each iteration is a
    substring of the wildtype sequence.

    Note: wildtype sequence expected as a string. X expected as an array or list
    of one-hot encodings.

    Returns optimal untested variant (as a string). """

    X_decode = [helpers.decode_X(x) for x in X]
    baseline = "" # stores baseline untested variant to be returned
    wt = list(wt)

    for i in range(4): # vary amino acid in each position
        fixed = ''.join(wt) # list of 3 fixed amino acids in each iter thru wt

        # index of xs in X with fixed substring
        index = [j for j, x in enumerate(X_decode) if fixed[0:i] == x[0:i] \
            and fixed[i + 1:len(fixed)] == x[i + 1:len(x)]]
        # stores y values of x's in X with those 3 fixed amino acids
        ys = [y[j] for j in index]

        # takes first occurrence of index with maximum y value
        max_ind = np.where(ys==max(ys))[0][0]

        # store amino acid in position being varied in baseline
        baseline += X_decode[index[max_ind]][i]

    return baseline


def avail_aa(X):
    """ Takes in a library X and returns a list of tuples of the available amino
    acids at each position that can be added to the library from V.

    Used for the Greedy algorithm baseline. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    return [(aa, i) for i in range(4) for aa in amino_acids if (aa, i) not in X]


def objective(X, L, probs, n):
    """ Objective (negative of objective to be maximized) to be minimized. """
    return opt.obj_LHS(X, L, probs) - opt.obj_RHS(X, L, probs, n)


def greedy(probs, seed, n, L):
    """ Takes in probabilities, seed (the best prediction), batch size, and L
    to create baseline optimal library using the Greedy algorithm.
    The algo starts out with the seed then continues to add amino acids until
    obj stops increasing.

    Note: probs expected as a dictionary, and seed expected as list of tuples.

    Returns optimal untested library. """

    X = seed # library X starts with seed

    obj = objective(X, L, probs, n)
    aa = avail_aa(X) # determine available aa at each position of X

    while True:
        # lst of obj's for lib w each available aa added
        lst = [objective(X + [a], L, probs, n) for a in aa]

         # determine which aa minimizes obj
        index, obj_next = min(enumerate(lst), key=operator.itemgetter(1))
        if obj_next > obj: # if obj stops decreasing, exit
            break
        else:
            X.append(aa[index]) # add aa that minimizes obj to X
            obj = obj_next
            aa.remove(aa[index])

    return X, obj
