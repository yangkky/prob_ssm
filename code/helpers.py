import torch
import numpy as np
from torch import distributions as dist

import operator


from gptorch import kernels, models

def decode_X(X):
    """ Takes in one-hot encoding X and decodes it to
    return a string of four amino acids. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'

    pos_X = [i for i, x in enumerate(X) if x == 1.0] # positions of amino acids
    pos_X = [(p - 20 * i) for i, p in enumerate(pos_X)] # make sure indexing is same as in str amino_acids
    aa_X = [amino_acids[p] for i, p in enumerate(pos_X)] # amino acid chars in X
    return ''.join(aa_X)

def get_predictions(X_train, y_train, X_test, its=500):
    """
    Train GP regressor on X_train and y_train.
    Predict mean and std for X_test.
    Return P(y > y_train_max) as dictionary eg 'AGHU': 0.78
    NB: for X_test in X_train, P ~= 0
    Be careful with normalization

    Expects X_train, y_train, and X_test as np.arrays
    """

    ke = kernels.MaternKernel()
    mo = models.GPRegressor(ke)

    # make data into tensors
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(np.array(X_test))
    y_train_scaled = (np.array(y_train) - np.mean(np.array(y_train))) / np.std(np.array(y_train)) # scale y_train
    y_train_scaled = torch.Tensor(y_train_scaled.reshape(len(y_train_scaled), 1)).float() # .float()

    his = mo.fit(X_train, y_train_scaled, its=its) # fit model with training set

    # make predictions
    dic = {} # use dictionary to store probs
    ind = 0 # index for feeding in batches of X_test
    tau = y_train_scaled.max().float()

    for i in range(1000, len(X_test) + 1000, 1000):
        mu, var = mo.forward(X_test[ind:i]) # make predictions
        std = torch.sqrt(var.diag())
        mu = mu.squeeze()
        prob = 1 - dist.Normal(mu, std).cdf(tau) # compute probabilities for all means, stds

        for j, p in enumerate(prob):
            seq = decode_X(X_test[ind:i][j]) # decode one-hot to get string of seq
            dic[seq] = p # store prob for each seq

        ind = i

    return dic

def generate_V(L):
    """ Returns V: a list of tuples with every possible amino acid
    at each of the four positions. """

    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    return [(aa, i) for i in range(L) for aa in amino_acids]

def get_seed(probs):
    """ Takes in a dictionary of amino acids to probabilities as
    generated by the get_predictions() function, and returns the
    seed (the four amino acid seq with the best prediction, aka the
    highest probabilitiy).

    Returns a list of tuples representing the seed.

    Currently, 'SSSG' is the seed. """

    seq = max(probs.items(), key=operator.itemgetter(1))[0]
    return [(aa, i) for aa, i in zip(seq, range(4))]