# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np

from util import *


class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        rng = np.random.default_rng()
        self.weights = rng.standard_normal(dim)  # FIXME

    def train(self, inputs, targets, alpha=0.1, eps=20, live_plot=False):
        if live_plot:
            print(self.weights, flush=True)
            plot_decision(self.weights, inputs, targets, show=False)
            interactive_on()

        (count, _) = inputs.shape
        errors = []

        for ep in range(eps):
            print('Ep {:3d}/{}: '.format(ep+1, eps), end='')
            overall_error = 0  # Overall error for episode

            for idx in np.random.permutation(count):  # "for each idx in random order"
                x = add_bias(inputs[idx])  # FIXME - input with bias
                d = targets[idx]             # FIXME

                net = self.weights @ x # FIXME integer
                y = 1 if net >= 0 else 0             # FIXME integer

                e = d - y            # FIXME - error on current input
                overall_error += e**2 / 2.0

                self.weights += aplha * e * x  # FIXME

            errors.append(overall_error)
            print('E = {:.3f}'.format(overall_error))

            if live_plot:
                clear()
                plot_decision(self.weights, inputs, targets, show=False)
                redraw()

        if live_plot:
            plt.show(block=True)
            interactive_off()

        return errors
