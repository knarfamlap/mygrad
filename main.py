from core import Value

a = Value(4.0)
b = Value(2.0)

f = 3 * a + b**2
f.backward()  
print(b.grad)
