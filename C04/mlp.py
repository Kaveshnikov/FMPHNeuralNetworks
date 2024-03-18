# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from util import *
import numpy as np


class MLP:
    """
    Multi-Layer Perceptron (abstract base class)
    """

    def __init__(self, dim_in, dim_hid, dim_out):
        """
        Initialize model, set initial weights
        """

        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        rng = np.random.default_rng()
        self.W_hid = rng.standard_normal(self.dim_hid + 1).reshape(-1, 1)  # FIXME
        self.W_out = rng.standard_normal(self.dim_out).reshape(-1, 1)  # FIXME

    # Activation functions & derivations
    # (not implemented, to be overridden in derived classes)
    def f_hid(self, x):
        raise NotImplementedError

    def df_hid(self, x):
        raise NotImplementedError

    def f_out(self, x):
        raise NotImplementedError

    def df_out(self, x):
        raise NotImplementedError

    # Back-propagation
    def forward(self, x):
        """
        Forward pass - compute output of network
        x: single input vector (without bias, size=dim_in)
        """

        a = self.W_hid @ add_bias(x)  # FIXME net vector on hidden layer (size=dim_hid)
        h = self.f_hid(a)  # FIXME activation of hidden layer (without bias, size=dim_hid)
        b = self.W_out @ add_bias(h)  # FIXME net vector on output layer (size=dim_out)
        y = self.f_out(b)  # FIXME output vector of network (size=dim_out)

        return a, h, b, y

    def backward(self, x, a, h, b, y, d):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        """

        g_out = (d - y) @ self.df_out(b)  # FIXME
        g_hid = np.sum(b @ g_out, axis=0) @ self.df_hid(a)  # FIXME

        dW_out = 0  # FIXME
        dW_hid = 0  # FIXME

        return dW_hid, dW_out
