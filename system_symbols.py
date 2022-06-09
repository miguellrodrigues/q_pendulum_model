import sympy as sp

Jeq, Mp, lp, r, Bp, Beq, Kt, Km, Rm, Jp, g = sp.symbols('J_{eq} M_p l_p r B_{p} B_{eq} K_t K_m R_m J_p g')

Mp2 = Mp ** 2
lp2 = lp ** 2
r2 = r ** 2

a, b, c, d, G = sp.symbols('a b c d G')


def A(alpha, alpha_dot):
  E = a * c - ((b ** 2) * (sp.cos(alpha) ** 2))

  return sp.Matrix([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, ((b * d * sp.sin(alpha) * sp.cos(alpha)) / alpha) / E, -((Beq + G) * c) / E,
     -(b * Bp + b * c * sp.sin(alpha) * alpha_dot) / E],
    [0, ((a * d * sp.sin(alpha)) / alpha) / E, -(b * (G + Beq) * sp.cos(alpha)) / E,
     -((b ** 2) * alpha_dot * sp.sin(alpha) * sp.cos(alpha) + Bp) / E]
  ])


def B(alpha):
  E = a * c - b ** 2 * sp.cos(alpha) ** 2

  return sp.Matrix([
    [0],
    [0],
    [((c * Kt) / Rm) / E],
    [((b * Kt) / (Rm * sp.cos(alpha))) / E]
  ]).T


_a, a_dot = sp.symbols('alpha \dot{\\alpha}')

_A = A(_a, a_dot)
_B = B(_a)

sp.print_latex(_A)
print(' ')
sp.print_latex(_B)
