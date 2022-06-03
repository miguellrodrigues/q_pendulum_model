import sympy as sp


def line_a_b(a, b):
  x, y, x1, x2, y1, y2 = sp.symbols('x y x1 x2 y1 y2')

  f = ((y2 - y1) / (x2 - x1)) * (x - x1)
  z = y - y1

  res = sp.solve(f - z, y)[0]

  a_x = a[0]
  a_y = a[1]

  b_x = b[0]
  b_y = b[1]

  return res.subs({
    x1: a_x,
    y1: a_y,
    x2: b_x,
    y2: b_y
  })


# binary counting from  0 to 16
def binary_counting(n):
  res = []
  for i in range(n + 1):
    res.append(bin(i)[2:])
  return res

print(binary_counting(16))