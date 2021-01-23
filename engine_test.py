from mygrad.engine import Variable

a = Variable(4.0)
b = Variable(2.0)

f =  a + 2
grads = f.backward()
print(f)
print(grads[a])