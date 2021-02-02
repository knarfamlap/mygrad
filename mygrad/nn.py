import random
import numpy as np
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


class RNN(Module):

    def __init__(self, input_sz, hidden_sz, output_sz):
        self.hidden_sz = hidden_sz
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.i2h = Linear(input_sz + hidden_sz, hidden_sz)
        self.i2o = Linear(input_sz + hidden_sz, output_sz)
        self.softmax = LogSoftmax()

        self.layers = [self.i2h, self.i2o, self.softmax]

    def __call__(self, x, hidden):
        # concatenate the input and hidden column wise
        comb = np.concatenate((x, hidden), axis=1)
        hidden = self.i2h(comb)
        output = self.i2o(comb)
        output = self.softmax(output)

        return output, hidden

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def initHidden(self):
        return np.zeros((1, self.hidden_sz))

    def __repr__(self):
        return "RNN of [ input_sz={}, hidden_sz={}, output_sz={}]".format(self.input_sz, self.hidden_sz, self.output_sz)


class Sigmoid(Module):

    def __call__(self, x):
        return [F.sigmoid(x) for elem in x]


class ReLU(Module):

    def __call__(self, x):
        return [F.relu(x) for elem in x]


class Tanh(Module):

    def __call__(self, x):
        return [F.tanh(x) for elem in x]


class LogSoftmax(Module):

    def __call__(self, x):
        exp_sum = sum(F.exp(xi) for xi in x)
        return [F.log(F.exp(xi) / exp_sum) for xi in x]
