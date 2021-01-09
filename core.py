class Value:

    def __init__(self, data, children=(), op=''):
        self.data = data
        self.children = children
        self.op = op

        self.prev = set(children)
        self.backward = lamda: None
        self.grad = 0              
