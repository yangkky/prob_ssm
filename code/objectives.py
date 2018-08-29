import itertools
from collections import Counter

import numpy as np
import torch

import helpers

def obj_LHS(X, L, probs):
    """ Takes in library X, and probabilities.

    Expects X to be a list of tuples, and probs to be a dictionary.

    Returns LHS of objective to be maximized (a supermodular function):
    sum of probabilities. """

    N = helpers.get_N(X, L)
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

    N = helpers.get_N(X, L)
    if N == 0:
        return torch.tensor(0.0)
    return obj_LHS(X, L, probs) * (1 - 1 / N) ** n


# For testing on objective with binomial prob of success term removed
def obj_RHS_edit(X, L, probs, n):
    return 0

def objective(X, L, probs, n):
    """ Objective (negative of objective to be maximized) to be minimized. """
    return obj_LHS(X, L, probs) - obj_RHS(X, L, probs, n)

def sample_obj(lib, model, tau, seq_to_x, X_all, observed=[],
               its=1000, n=100, return_all=False):
    num_greater = torch.zeros(its)
    lib = helpers.seqs_from_set(lib, 4)
    unseen_lib = np.array(sorted(set(lib) - set(observed)))
    if len(unseen_lib) > 1:
        X_test = torch.tensor(X_all[[seq_to_x[s] for s in unseen_lib]]).float()
        mu, K = model(X_test)

        for i in range(its):
            rand_inds = np.random.choice(len(lib), n, replace=True)
            rand_inds = np.unique(rand_inds)
            rand_inds = rand_inds[np.where(rand_inds < len(unseen_lib))]
            if len(rand_inds) < 1:
                continue
            else:
                mu_chosen = mu[rand_inds].squeeze()
                K_chosen = K[rand_inds][:, rand_inds]
                if len(rand_inds) == 1:
                    sample = dist.Normal(mu_chosen,
                                         torch.sqrt(K_chosen.squeeze)).sample()
                else:
                    sample = dist.MultivariateNormal(mu_chosen, K_chosen).sample()
            num_greater[i] = torch.sum(sample > tau)
    if return_all:
        return -torch.mean(num_greater), num_greater
    else:
        return -torch.mean(num_greater)
