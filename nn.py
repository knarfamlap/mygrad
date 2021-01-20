import random
import numpy as np
from core import Value


class Module:

    def zero_grad(self):
        """
        Initialize all gradients to 0
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        """
        Initialize parameters for linear layer
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        # take the dot product of w and x and add the bias: wx + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # apply non linearity if indicated
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Returns the parameters of the layer
        """
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1)
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class RNN(Module):

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.i2h = Layer(input_size + hidden_size, hidden_size)
        self.i2o = Layer(input_size + hidden_size, output_size)

    def __call__(self, x, hidden):
        combined = np.concatenate((X, hidden), axis=1)
        hidden = self.i2h(combined)
        out = self.i2o(combined)
        return out, hidden

    def layers(self):
        return [self.i2h, i2o]

    def initHidden(self):
        return [[0 for _ in range(self.hidden_size)]]

    def parameters(self):
        return [p for layer in self.layers() for p in layer.parameters()]

    def __repr__(self):
        return f"RNN of [{', '.join(str(layer) for layer in self.layers())}]"
