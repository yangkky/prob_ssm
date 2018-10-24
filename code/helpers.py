import torch
import numpy as np
from torch import distributions as dist

import operator
import itertools
from collections import Counter

from gptorch import kernels, models

def decode_X(X):
    """ Takes in one-hot encoding X and decodes it to
    return a string of four amino acids. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'

    pos_X = [i for i, x in enumerate(X) if x == 1.0] # positions of amino acids
    pos_X = [(p - 20 * i) for i, p in enumerate(pos_X)] # make sure indexing is same as in str amino_acids
    aa_X = [amino_acids[p] for i, p in enumerate(pos_X)] # amino acid chars in X
    return ''.join(aa_X)

def encode_X(X):
    """ Takes in a string of four amino acids and encodes it
    to return a one-hot encoding. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'

    enc = np.array([0.] * 80)
    pos_X = [amino_acids.find(char) for char in X] # positions of amino acids
    for i, pos in enumerate(pos_X):
        enc[pos + i * 20] = 1.0
    return enc

def seqs_from_set(chosen, L):
    """ Takes in a list of tuples representing the library chosen, and
    the length L of the sequences made from that library. Returns the
    list of sequences from that library. """

    pos = [[c[0] for c in chosen if c[1] == p] for p in range(L)]
    return [''.join(s) for s in itertools.product(*pos)]

def get_predictions(X_train, y_train, X_test, one_hots=None, lr=1e-1,
                    sn=0.1, its=500, return_model=False):
    """
    Train GP regressor on X_train and y_train.
    Predict mean and std for X_test.
    Return P(y > y_train_max) as dictionary eg 'AGHU': 0.78, and predictions (means) as list
    NB: for X_test in X_train, P ~= 0
    Be careful with normalization

    Expects X_train, y_train, and X_test as np.arrays
    """

    ke = kernels.MaternKernel()
    mo = models.GPRegressor(ke, sn=sn, lr=lr, prior=False)

    # make data into tensors
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(np.array(X_test))
    y_train_scaled = (np.array(y_train) - np.mean(np.array(y_train))) / np.std(np.array(y_train)) # scale y_train
    y_train_scaled = torch.Tensor(y_train_scaled.reshape(len(y_train_scaled), 1)) # .float()

    his = mo.fit(X_train, y_train_scaled, its=its) # fit model with training set

    # make predictions
    dic = {} # use dictionary to store probs
    ind = 0 # index for feeding in batches of X_test
    tau = y_train_scaled.max()

    means = {} # use dictionary to store means

    if one_hots is None:
        one_hots = X_test
    for i in range(1000, len(X_test) + 1000, 1000):
        mu, var = mo.forward(X_test[ind:i]) # make predictions
        std = torch.sqrt(var.diag())
        mu = mu.squeeze()
        prob = 1 - dist.Normal(mu, std).cdf(tau) # compute probabilities for all means, stds

        for j, p in enumerate(prob):
            seq = decode_X(one_hots[ind:i][j]) # decode one-hot to get string of seq
            dic[seq] = p # store prob for each seq
            means[seq] = mu[j]

        ind = i
    if return_model:
        return dic, means, mo
    else:
        return dic, means

def get_mean_abs_err(X, y, mu, lib):
    """ Takes in X, true y values, predictions mu, and the sample X's (library)
    that the model was trained on, and returns list of abs errors for all y's
    not trained on and mean abs error.

    Expects X as one-hot encodings, y and mu as lists of floats, and
    lib as list of strings of four aa seqs.

    Returns a tuple of y_test (does not include y's corresponding to sample X's)
    and abs errors, and the mean abs error. """

    str_x = [decode_X(x) for x in X]
    inds = [i for i, x in enumerate(str_x) if x in lib] # indices of each seq in lib in X

    y_test = list(y) # remove corresponding y's and mu's of seqs in lib
    mu_test = mu.copy()
    for i in inds:
        y_test.pop(i)
        mu_test.pop(i)

    errs = [abs(mu - y).item() for mu, y in zip(mu_test, y_test)]
    return (y_test, errs), np.mean(np.array(errs))

def generate_V(L):
    """ Returns V: a list of tuples with every possible amino acid
    at each of the four positions. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    return [(aa, i) for i in range(L) for aa in amino_acids]

def get_N(X, L):
    """ Takes in library X and length L, and returns the library size N. """
    N = 1 # represents the product of sequence of # aas at each position
    counts = Counter([x[1] for x in X])
    for i in range(L):
        N *= counts[i]
    return N

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

def get_seed(probs):
    """ Takes in a dictionary of amino acids to probabilities as
    generated by the get_predictions() function, and returns the
    seed (the four amino acid seq with the best prediction, aka the
    highest probabilitiy).

    Returns a list of tuples representing the seed.

    Currently, 'SSSG' is the seed. """

    seq = max(probs.items(), key=operator.itemgetter(1))[0]
    L = len(list(probs.items())[0][0])
    return [(aa, i) for aa, i in zip(seq, range(L))]

def avail_aa(X):
    """ Takes in a library X and returns a list of tuples of the available amino
    acids at each position that can be added to the library from V.

    Used for the Greedy algorithm baseline. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    return [(aa, i) for i in range(4) for aa in amino_acids if (aa, i) not in X]
