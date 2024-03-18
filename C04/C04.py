#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from regressor import *
from util import *
import numpy as np


I_WANT_3D_PLOTS = False

# # Load data
data = np.loadtxt('mhd-easy.dat')
# Prune empty cells (only works if you use @2D plots)
if not I_WANT_3D_PLOTS:
    data = data[:, data[-1] != 0]

inputs = data[0:2]
targets = data[2:]  # keep the regression targets as a 2D-matrix with 1 column


# # Normalize inputs
frob_norm = np.linalg.norm(inputs, 'fro')
#inputs -= 0   # FIXME: move to zero-mean
inputs /= frob_norm   # FIXME: scale to unit-variance

#targets -= 0  # FIXME: move to zero-mean
frob_norm = np.linalg.norm(targets, 'fro')
targets /= frob_norm  # FIXME: scale to unit-variance


# # Train & visualize
# Build model
model = MLPRegressor(dim_in=inputs.shape[0], dim_hid=20, dim_out=targets.shape[0])
# Train model
trainREs = model.train(inputs, targets, alpha=0.05, eps=100)

# "Test" model
outputs = model.predict(inputs)
# Visualize
plot_reg_density('Density', inputs, targets, outputs, block=False, plot_3D=I_WANT_3D_PLOTS)
plot_errors('Model loss', trainREs, block=False)
