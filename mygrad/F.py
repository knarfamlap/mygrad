from mygrad.engine import Variable
import numpy as np


def sigmoid(x: Variable):
    return 1 / (1 + exp(-x))


def exp(x: Variable):
    value = np.exp(x.value)
    local_gradients = (
        (x, value)
    )

    return Variable(value, local_gradients)


def relu(x: Variable):
    value = 0 if x.value < 0 else x.value
    local_gradients = (
        (x, value)
    )

    return Variable(value, local_gradients)


def tanh(x: Variable):
    value = np.tanh(x.value)
    local_gradients = (
        (x, 1 - np.tanh(x)**2)
    )

    return Variable(value, local_gradients)

