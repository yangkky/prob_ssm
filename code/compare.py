import itertools
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from torch import distributions as dist

from gptorch import kernels, models

import helpers, opt, objectives

def greedy(rounds, params):
    train_inds, A, y, A_test, V, X, n, L = params
    libraries = []
    histories = []
    observed_inds = [train_inds]

    for rou in range(rounds):
        inds = sorted(set(itertools.chain(*observed_inds)))
        dic, _ = helpers.get_predictions(A[inds], y[inds], A_test,
                                         one_hots=X, its=3000, lr=1e-3)
        print()
        seen_seqs = [helpers.decode_X(X[i]) for i in inds]
        for s in seen_seqs:
            dic[s] = 0.0
        X0 = helpers.get_seed(dic)
        chosen, obj = opt.greedy(V, X0, objectives.objective,
                                 obj_args=(L, dic, n))
        print(obj)
        libraries.append(chosen)
        histories.append(obj)
        seqs = helpers.seqs_from_set(chosen, L)
        inds = np.random.choice(len(seqs), n, replace=True)
        sampled_seqs = [seqs[i] for i in inds]
        inds = [seq_to_x[s] for s in sampled_seqs]
        observed_inds.append(inds)

    return libraries, histories, observed_inds


if __name__ == '__main__':
    np.random.seed(120120)
    _ = torch.manual_seed(43298)
    L = 4
    n = 100
    rounds = 3

    parser = argparse.ArgumentParser()
    # Input file
    parser.add_argument('--input', required=True)
    # output file
    parser.add_argument('--output', required=True)
    # Replicates
    parser.add_argument('--repeats', default=2)
    args = parser.parse_args()

    # Load the inputs
    with open(args.input, 'rb') as f:
        t = pickle.load(f)
    X = t[0]
    A = t[1]
    y = t[2]
    wt = t[3]
    aas = 'ARNDCQEGHILKMFPSTWYV'

    seq_to_x = {}
    for i, x in enumerate(X):
        seq = helpers.decode_X(x)
        seq_to_x[seq] = i

    ground = [(aa, i) for aa in aas for i in range(L)]
    wt_inds = [seq_to_x[wt]]

    results = {
            'greedy': {},
            'modmod': {}
        }

    for rep in range(args.repeats):
        print('Replication %d' %(rep + 1))
        train_inds = wt_inds + list(np.random.choice(len(X), n, replace=True))

        y_train = y[train_inds]
        y_true = y
        A_train = A[train_inds]
        A_test = A

        params = train_inds, A, y, A_test, ground, X, n, L

        print('Greedy')
        glibs, ghists, ginds = greedy(rounds, params)
        # libraries = []
        # histories = []
        # observed_inds = [train_inds]
        #
        #
        # for rou in range(rounds):
        #     inds = sorted(set(itertools.chain(*observed_inds)))
        #     dic, _ = helpers.get_predictions(A[inds], y[inds], A_test,
        #                                      one_hots=X, its=3000, lr=1e-3)
        #     print()
        #     seen_seqs = [helpers.decode_X(X[i]) for i in inds]
        #     for s in seen_seqs:
        #         dic[s] = 0.0
        #     seed = helpers.get_seed(dic)
        #     chosen, obj = opt.greedy(objectives.objective, seed,
        #                              obj_args=(n, dic, L))
        #     libraries.append(chosen)
        #     histories.append(obj)
        #     seqs = helpers.seqs_from_set(chosen, L)
        #     inds = np.random.choice(len(seqs), n, replace=True)
        #     sampled_seqs = [seqs[i] for i in inds]
        #     inds = [seq_to_x[s] for s in sampled_seqs]
        #     observed_inds.append(inds)
