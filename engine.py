from collections import defaultdict


class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other):
        other = Variable(other) if isinstance(other, (int, float)) else other
        value = self.value + other.value
        local_gradients = (
            (self, 1),
            (other, 1)
        )

        return Variable(value, local_gradients=local_gradients)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = Variable(other) if isinstance(other, (int, float)) else other

        value = self.value * other.value
        local_gradients = (
            (self, other.value),
            (other, self.value)
        )

        return Variable(value, local_gradients)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self) 

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __neg__(self):
       return self * -1

    def __pow__(self, val):
        assert isinstance(val, (int, float))
        value = self.value ** val
        local_gradients = (
            (self, val * self.value ** (val - 1)),
        )

        return Variable(value, local_gradients)

    def __repr__(self):
        return "Variable(value={})".format(self.value)

    def backward(self):
        grads = defaultdict(lambda: 0)

        def compute_grads(variable, path_value):
            for child_variable, local_gradient in variable.local_gradients:
                value_of_path_to_child = path_value * local_gradient
                grads[child_variable] += value_of_path_to_child
                compute_grads(child_variable, value_of_path_to_child)

        compute_grads(self, path_value=1)

        return grads
