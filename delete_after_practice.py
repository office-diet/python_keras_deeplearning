def generator():
  i = 0
  j = i + 1
  while True:
    i += 1
    j += 1
    yield i, j

gen = generator()

x, y = gen.__next__()
print(x, y)
x, y = gen.__next__()
print(x, y)
x, y = gen.__next__()
print(x, y)
