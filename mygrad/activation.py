from mygrad.engine import Variable
from mygrad.nn import Module
import mygrad.F as F 



class Sigmoid(Module):

    def __call__(self, x):
        return [F.sigmoid(x) for elem in x]

class ReLU(Module):

    def __call__(self, x):
        return [F.relu(x) for elem in x]

class Tanh(Module):

    def __call__(self, x):
        return [F.tanh(x) for elem in x]


