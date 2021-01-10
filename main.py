from core import Value

a = Value(4)
b = Value(2)


f = 3 * a + b**3
c = f.relu()

print(c.data)

d = f.sigmoid()

print(d.data)

f.backward()
dfb = b.grad

print(dfb.data)
dfb.backward()
dfbb = b.grad

print(dfbb.data)
dfbb.backward()
dfbbb = b.grad

print(dfbbb.data)
dfbbb.backward()
