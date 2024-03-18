# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from mlp import *
from util import *


class MLPRegressor(MLP):
    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__(dim_in, dim_hid, dim_out)

    # Activation functions & derivations

    def error(self, targets, outputs):
        """
        Cost / loss / error function
        """

        return np.sum((targets - outputs)**2, axis=0)

    # @override
    def f_hid(self, x):
        return 0  # FIXME: sigmoid

    # @override
    def df_hid(self, x):
        return 0  # FIXME: derivation of sigmoid

    # @override
    def f_out(self, x):
        return 0  # FIXME: linear

    # @override
    def df_out(self, x):
        return 0  # FIXME: derivation of linear

    def predict(self, inputs):
        """
        Prediction = forward pass
        """

        # If self.forward() can process only one input at a time
        outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        # # If self.forward() can take a whole batch
        # *_, outputs = self.forward(inputs)
        return outputs

    def train(self, inputs, targets, alpha=0.1, eps=100):
        """
        Training of the regressor
        inputs: matrix of input vectors (each column is one input vector)
        targets: matrix of target vectors (each column is one target vector)
        alpha: learning rate
        eps: number of episodes
        """

        (_, count) = inputs.shape
        errors = []

        for ep in range(eps):
            E = 0

            for idx in np.random.permutation(count):
                x = 0  # FIXME
                d = 0  # FIXME

                a, h, b, y = self.forward(x)
                dW_hid, dW_out = self.backward(x, a, h, b, y, d)

                self.W_hid += 0  # FIXME
                self.W_out += 0  # FIXME

                E += self.error(d, y)

            E /= count
            errors.append(E)
            if (ep+1) % 5 == 0:
                print('Epoch {:3d}/{}, E = {:.3f}'.format(ep+1, eps, E))

        return errors
