from mygrad.engine import Variable
import mygrad.F as F
import torch
import torch.nn.functional as Fpt

def test_ops():
    a = Variable(2.0)
    b = Variable(3.0)

    d = a * b + b**3
    c = d - a
    c += a
    g = c / a
    f = -g 

    f.backward()
    
    amg, bmg, fval = a.grad, b.grad,  f.value

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

    assert abs(amg - apt.grad.item()) < tol
    assert abs(bmg - bpt.grad.item()) < tol
    assert abs(fval - fpt.data.item()) < tol


def test_sigmoid():
    a = Variable(5.0)
    b = Variable(3.0)

    c = a + b
    # d = c ** 2
    e = F.sigmoid(c)

    e.backward()
    amg, bmg, emg_val = a.grad, a.grad, e.value

    a = torch.Tensor([5.0]).double()
    b = torch.Tensor([3.0]).double()

    a.requires_grad = True
    b.requires_grad = True

    c = a + b
    # d = c ** 2
    e = torch.sigmoid(c)

    e.backward()
    apt, bpt, ept_val =  a, b, e

    tol = 1e-6

    assert abs(amg - apt.grad.item()) < tol
    assert abs(bmg - bpt.grad.item()) < tol
    assert abs(emg_val - ept_val.data.item()) < tol


def test_relu():
    a = Variable(5.0)
    b = Variable(3.0)

    c = a + b
    d = c ** 2
    e = F.relu(d)

    e.backward()
    amg, bmg, emg_val = a.grad, b.grad, e.value

    a = torch.Tensor([5.0]).double()
    b = torch.Tensor([3.0]).double()

    a.requires_grad = True
    b.requires_grad = True

    c = a + b
    d = c ** 2
    e = Fpt.relu(d)

    e.backward()
    apt, bpt, ept_val =  a, b, e

    tol = 1e-6

    print("mygrad: {}, pytorch: {}".format(amg, apt.grad.item()))
    assert abs(amg - apt.grad.item()) < tol
    assert abs(bmg - bpt.grad.item()) < tol
    assert abs(emg_val - ept_val.data.item()) < tol


def test_tanh():
    a = Variable(5.0)
    b = Variable(3.0)

    c = a + b
    d = c ** 2
    e = F.tanh(d)

    e.backward()
    amg, bmg, emg_val = a.grad, b.grad, e.value

    a = torch.Tensor([5.0]).double()
    b = torch.Tensor([3.0]).double()

    a.requires_grad = True
    b.requires_grad = True

    c = a + b
    d = c ** 2
    e = torch.tanh(d)

    e.backward()
    apt, bpt, ept_val =  a, b, e

    tol = 1e-6

    assert abs(amg - apt.grad.item()) < tol
    assert abs(bmg - bpt.grad.item()) < tol
    assert abs(emg_val - ept_val.data.item()) < tol
