# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2024

from mlp import *
from util import *


class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid, n_classes):
        self.n_classes = n_classes
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    def error(self, targets, outputs):
        """
        Cost / loss / error function
        """

        return np.sum((targets - outputs)**2, axis=0)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))    

    # @override
    def f_hid(self, x):
        return self.sigmoid(x)  # FIXME sigmoid (or tanh, relu, ...)

    # @override
    def df_hid(self, x):
        return 0  # FIXME corresponding derivation

    # @override
    def f_out(self, x):
        return self.sigmoid(x)  # FIXME sigmoid

    # @override
    def df_out(self, x):
        return   # FIXME corresponding derivation

    def predict(self, inputs):
        """
        Prediction = forward pass
        """

        # If self.forward() can process only one input at a time
        outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        # # If self.forward() can take a whole batch
        # *_, outputs = self.forward(inputs)
        return outputs, onehot_decode(outputs)

    def test(self, inputs, labels):
        """
        Test model: forward pass on given inputs, and compute errors
        """
        targets = 0  # FIXME: convert labels to target vectors (each column is one target vector)
        outputs, predicted = self.predict(inputs)
        CE = 0       # FIXME: mean classification error (a number from range <0, 1>)
        RE = 0       # FIXME: regression error
        return CE, RE

    def train(self, inputs, labels, alpha=0.1, eps=100, live_plot=False, live_plot_interval=10):
        """
        Training of the classifier
        inputs: matrix of input vectors (each column is one input vector)
        labels: vector of labels (each item is one class label)
        alpha: learning rate
        eps: number of episodes
        live_plot: plot errors and data during training
        live_plot_interval: refresh live plot every N episodes
        """
        (_, count) = inputs.shape
        targets = 0  # FIXME: convert labels to target vectors (each column is one target vector)

        if live_plot:
            interactive_on()

        CEs = []
        REs = []

        for ep in range(eps):
            CE = 0
            RE = 0

            for idx in np.random.permutation(count):
                x = inputs[:, idx]
                d = targets[:, idx]

                a, h, b, y = self.forward(x)
                dW_hid, dW_out = self.backward(x, a, h, b, y, d)

                self.W_hid += alpha * dW_hid
                self.W_out += alpha * dW_out

                CE += labels[idx] != onehot_decode(y)
                RE += self.error(d, y)

            CE /= count
            RE /= count
            CEs.append(CE)
            REs.append(RE)
            if (ep+1) % 5 == 0: print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep+1, eps, CE, RE))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                _, predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                plot_areas(self, inputs, block=False)
                redraw()

        if live_plot:
            interactive_off()

        print()

        return CEs, REs
