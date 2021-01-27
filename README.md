# mygrad

mygrad is  [micrograd](https://github.com/karpathy/micrograd) with a couple of extra features. The extra features are some 
non-linear function and high degree gradients. Currently its only a scalar value autograd engine but 
I hope to implement Tensors and the like. The intention of this repo is simply for the sake of my 
understanding.

## Installation

First clone the repo
```
$ git clone https://github.com/knarfamlap/mygrad.git
```

Then install the dependencies for the examples and for some functions to work
``` 
$ python3 install -r requirements.txt
```

## Examples

### Simple Operations

Addition, Subtraction, Multiplication and Division is currently supported.

```python
a = Variable(2.0)
b = Variable(2.0)

c = (a + b) / 2 # Result: 2.0

d = c ** 3 # Result: 8.0

f = -c # Result: -8.0 
```

### Non-Linearities

Sigmoid, ReLU, Tanh, and Exp are also supported

``` python
a = Variable(2.0)
b = Variable(2.0)

c = F.exp(a) * F.exp(b) # Result: 54.5982

d = F.relu(a + b) # Result: 4.0

f = F.tanh(a * b) # Result: 0.9993

w = F.sigmoid(a ** 3) # Result: 0.9996
```

### First Order Derivative

You can easily take the first order derivative of functions

``` python
a = Variable(2.0)
b = Variable(3.0)

f = a ** 2 + b ** 2 # Result: 13.0 

f.backward() # backward pass 

print(a.grad) # Result: 4.0
print(b.grad) # Result: 6.0
```
