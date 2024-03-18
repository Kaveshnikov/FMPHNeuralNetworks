#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np
import matplotlib.pyplot as plt
from util import *


## Generate data
# number of points
N = 100

# generate x values
x = np.arange(N).reshape(-1, 1)   # FIXME: vector [0, 1, 2, ..., N-1]

# generate random slope (k) and shift (q)
k = np.random.rand()   # FIXME: random number from normal distribution
q = np.random.randn()   # FIXME: random number from uniform distribution

# calculate y values
y = k * x + q   # FIXME: according to slides

# add noise
rng = np.random.default_rng()
y += rng.standard_normal(y.size).reshape(-1, 1)  # FIXME: add vector of random values from normal distribution

# plot all points
plt.scatter(x, y)


## Find k and q using linear regression
# first append ones to x
X = np.concatenate((x, np.ones((x.size, 1))), axis=1)  # FIXME: X should be matrix of two columns: first column is vector x, second column are ones

# then find params
k_pred, q_pred = np.inv(X.T @ X).dot(X.T).dot(y)   # FIXME: according to slides

# predict ys and plot as a line
y_pred = k_pred * x + q_pred   # FIXME: according to slides

# plot predicted
plt.plot(x, y_pred, 'r')
use_keypress(plt.gcf())   # Use ' ' or 'enter' to close figure; 'q' or 'escape' to quit program
plt.show()

