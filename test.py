import torch

from core import Value


def test_relu():
    x = Value(-2.0)
    f = 3 * x + x**2
    h = f.relu()

    h.backward()

    xmg = x

    x = torch.Tensor([-2.0]).double()
    x.requires_grad = True
    f = 3 * x + x**2
    h = f.relu()

    h.backward()

    xpt = x

    # forward
    assert xmg.data == xpt.data.item()
    # backward
    assert xmg.grad.data == xpt.grad.item()


def test_sigmoid():
    x = Value(-2.0)
    f = 3 * x 
    h = f.sigmoid()

    h.backward()

    xmg = x

    x = torch.Tensor([-2.0]).double()
    x.requires_grad = True
    f = 3 * x
    h = f.sigmoid()

    h.backward()

    xpt = x
    print("torch sigmoid: {} \n mygrad sigmoid: {}".format(xpt.data.item(), xmg.data))
    # forward
    assert xmg.data == xpt.data.item()
    # backward
    print("torch sigmoid: {} \n mygrad sigmoid: {}".format(xpt.grad.item(), xmg.grad.data))
    assert xmg.grad.data == xpt.grad.item()


def test_tanh():
    x = Value(-4.0)
    f = 3 * x / (x**5)
    h = f.tanh()

    h.backward()

    xmg = x

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    f = 3 * x / (x ** 5)
    h = f.tanh()

    h.backward()

    xpt = x

    # forward
    assert xmg.data == xpt.data.item()
    # backward
    print("torch tanh: {} \n mygrad tanh: {}".format(xpt.grad.item(), xmg.grad.data))
    assert xmg.grad.data == xpt.grad.item()


# print("Testing Relu")
# test_relu()
# print("Relu Test Passed")

# print("Testing Sigmoid")
# test_sigmoid()
# print("Sigmoid Test Passed")

print("Testing Tanh")
test_tanh()
print("Tang Test Passed")
