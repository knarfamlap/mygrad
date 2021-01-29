import random
import mygrad.F as F
from mygrad.engine import Variable


class Module:

    def zero_grad(self):
        for p in self.parameters():
            # set all gradients to zeros
            p.grad = 0
            pass

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, inputs, nonlin=True):
        self.w = [Variable(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Variable(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        return F.relu(act) if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return 'Neuron({})'.format(len(self.w))


class Linear(Module):
    def __init__(self, inputs, outputs, **kwargs):
        self.neurons = [Neuron(inputs, **kwargs) for _ in range(outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Linear of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i != len(nouts)-1)
                       for i in range(len(nouts))]  # no nonlin on last layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
