from core import Value

a = Value(4)
b = Value(2)


f = 3 * a + b**2

f.backward()
dfb = b.grad

print(dfb.data)
dfb.backward()
dfbb = b.grad

print(dfbb.data)
dfbb.backward()
dfbbb = b.grad

print(dfbbb.data)



