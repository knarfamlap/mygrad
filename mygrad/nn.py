import random
from mygrad.engine import Variable


class Module:

    def zero_grad(self):
        for p in self.parameters():
            # set all gradients to zeros
            p.local_gradients = ()
            pass

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, inputs):
        self.w = [Variable(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Variable(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return 'Neuron({})'.format(len(self.w))


class Linear(Module):
    def __init__(self, inputs, outputs):
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return "Linear of [{}]".format(str(n) + ',' for n in self.neurons)


