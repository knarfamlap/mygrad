from mygrad.engine import Variable
import torch

def test_ops():
    a = Variable(2.0)
    b = Variable(3.0)

    d = a * b + b**3
    c = d - a
    c += a
    g = c / a
    f = -g 

    grads = f.backward()
    
    amg, bmg, fval = grads[a], grads[b], f.value

    a = torch.Tensor([2.0]).double()
    b = torch.Tensor([3.0]).double()

    a.requires_grad = True
    b.requires_grad = True

    d = a * b + b**3
    c = d - a
    c += a 
    g = c / a
    f = -g 

    f.backward()
    
    apt, bpt, fpt = a, b, f

    tol = 1e-6

    print("mygrad: {}, pytorch: {}".format(fval, fpt.data.item()))
    assert abs(amg - apt.grad.item()) < tol
    assert abs(bmg - bpt.grad.item()) < tol
    assert abs(fval - fpt.data.item()) < tol


    