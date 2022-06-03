import sympy as sp
import numpy as np


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


p1 = [sp.rad(20) -sp.pi/4, -1]
p2 = [-sp.rad(20) + sp.pi/4, 1]

line = line_a_b(p1, p2)
print(line)
