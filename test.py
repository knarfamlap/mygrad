import torch
import numpy as np
from core import Value

def test_exp():
    a = Value(4.0)
    b = Value(2.0)

    x = a 

    x = x.exp()


    x.backward()
    print(a.grad)
    assert a.grad == np.exp(4.0) 

# print("Testing Relu")
# test_relu()
# print("Relu Test Passed")

# print("Testing Sigmoid")
# test_sigmoid()
# print("Sigmoid Test Passed")

test_exp()
