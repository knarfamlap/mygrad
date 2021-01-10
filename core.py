from math import exp

class Value:

    def __init__(self, data, children=(), op=''):
        self.data = data
        self.children = children
        self.op = op

        self._prev = set(children)
        self._backward = lambda: None
        self.grad = 0


    def __add__(self, other):
        other = self.assign_other(other) 
        out = Value(self.data + other.data, (self, other), '+') # the output is a Value obj
        # defines the backward function of add
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward # assign the function to the attribute

        return out

    def __mul__(self, other):
        other = self.assign_other(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other * out.grad
            other.grad += self * out.grad

        out._backward = _backward

        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), '**{}'.format(other))  
        
        def _backward():
            self.grad += (other * self**(other-1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        def _sigmoid_fn(x):
            return 1 / (1 + exp(-x))
        
        out = Value(_sigmoid_fn(self.data), (self, ), 'Sigmoid')

        def _backward():
            self.grad += (1 - _sigmoid_fn(out.data) * _sigmoid_fn(out.data)) * out.grad

        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                v.grad = 0 # grad to 0 so calculation is not affected by previous results
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # go one var at a time and apply chain rule to get its gradient
        self.grad = Value(1)
        for v in reversed(topo):
            v._backward()


    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __repr__(self):
        return "Value(data={}, grad={})".format(self.data, self.grad) 

    def assign_other(self, other):
        return other if isinstance(other, Value) else Value(other)
