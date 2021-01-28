from mygrad.engine import Variable
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


def exp(x):
    value = np.exp(x.value)
    local_gradients = (
        (x, value),
    )

    return Variable(value, local_gradients)


def relu(x):
    value = 0 if x.value < 0 else x.value
    local_gradients = (
        (x,  (value > 0) * 1),
    )

    return Variable(value, local_gradients)


def tanh(x):
    value = np.tanh(x.value)
    local_gradients = (
        (x,  1. - value**2),
    )

    return Variable(value, local_gradients)
