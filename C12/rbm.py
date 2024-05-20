# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017 - 2024

import numpy as np

from util import *


## Helpers

def sample(p):
    '''
    Sample x_i from {0,1} with probability p_i (Bernoulli)
    p: vector of probabilities of length N
    returns: y - vector of length N
    '''
    return np.zeros(p.shape)  # FIXME



class RBM():
    '''
    Restricted Boltzmann Machine
    '''

    def __init__(self, dim_vis, dim_hid, inputs=None):
        self.dim_vis = dim_vis
        self.dim_hid = dim_hid

        self.W = 0.01 * 0  # FIXME
        self.a = 0.01 * 0  # FIXME
        self.b = 0.01 * 0  # FIXME

        # Initialize visible biases by a heuristic
        if inputs is not None:
            x = np.mean(inputs, axis=1) # Proportion of 1s for each input pixel, i.e. P[actual_input_i == 1]
            epsilon = 0.0001
            x = np.clip(x, epsilon, 1-epsilon)
            self.a = np.log(x / (1-x))  # Now P[predicted_input_i == 1 | random_hidden] == P[actual_input_i == 1]



    def f(self, x):
        '''
        "Activation function" - logistic sigmoid
        Technically, this is not an activation function, but f: net -> probability.
        '''
        return 1 / (1 + np.exp(-x))



    def forward(self, v, use_probs=False):
        '''
        Sample a probable hidden state from the visible state
        use_probs: if True: return only probability vector p
                   if False: use probability vector p and sample activations (true forward pass)
        '''
        p = np.zeros(self.dim_hid)  # FIXME
        return p if use_probs else sample(p)


    def backward(self, h, use_probs=False):
        '''
        Sample a probable visible state from the hidden state
        use_probs: if True: return only probability vector p
                   if False: use probability vector p and sample activations (true forward pass)
        '''
        p = np.zeros(self.dim_vis)  # FIXME
        return p if use_probs else sample(p)



    def gibbs(self, h, rounds, use_probs=False):
        '''
        Sample a probable visible state from the hidden state using Gibb`s sampling
        use_probs: if True: return only probability vector p
                   if False: use probability vector p and sample activations (true forward pass)
        '''
        for _ in range(rounds):
            v = 0                      # FIXME
            h = 0                      # FIXME

        return np.zeros(self.dim_vis)  # FIXME



    def cd(self, v_pos, gibbs_rounds=0, use_probs=False):
        '''
        Compute Contrastive Divergence gradients for W, a, b, and error e
        Use Gibb`s sampling for estimate of v_neg
        '''
        h_pos = 0  # FIXME
        v_neg = 0  # FIXME
        h_neg = 0  # FIXME

        dW = 0     # FIXME
        da = 0     # FIXME
        db = 0     # FIXME

        e = np.linalg.norm(v_pos - self.backward(h_pos))
        return dW, da, db, e



    def train(self, inputs, alpha=0.1, eps=100, gibbs_rounds=0, use_probs=False):
        '''
        Train model using traditional SGD
        '''
        (_, count) = inputs.shape

        for ep in range(eps):
            E = 0
            for t in np.random.permutation(count):
                x = inputs[:,t]

                dW, da, db, e = self.cd(x, gibbs_rounds=gibbs_rounds, use_probs=use_probs)

                self.W += alpha * dW
                self.a += alpha * da
                self.b += alpha * db
                E += e

            if (ep+1) % 10 == 0:
                print('  Ep {:3d}/{}: E = {:.3f}'.format(ep+1, eps, E))



    def generate(self, gibbs_rounds=0):
        '''
        Generate a probable visible state from random a hidden state
        '''
        h = (np.random.rand(self.dim_hid) < 0.5).astype(float)
        return self.gibbs(h, rounds=gibbs_rounds)
