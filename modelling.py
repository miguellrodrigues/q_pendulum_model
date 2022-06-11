import sympy as sp


# E = T + V
# L = T - V

kt, km, vm, rm = sp.symbols('K_t K_m V_m R_m')
theta, alpha = sp.symbols('theta alpha', cls=sp.Function)
Mp, g, lp, r = sp.symbols('M_p g l_p r')
Jeq, Jp = sp.symbols('J_eq J_p')
Beq, Bp = sp.symbols('B_eq B_p')

t = sp.symbols('t')

theta = theta(t)
theta_dot = sp.diff(theta, t)
d2_theta = sp.diff(theta_dot, t)

alpha = alpha(t)
alpha_dot = sp.diff(alpha, t)
d2_alpha = sp.diff(alpha_dot, t)

tal = (kt*vm - kt*km*theta_dot)/rm

vx = r*theta_dot - lp*alpha_dot*sp.cos(alpha)
vy = -lp*alpha_dot*sp.sin(alpha)

# Potential energy
V = Mp*g*lp*sp.cos(alpha)

# Kinetic energy
# T = Tpend_rot + Tarm_rot + Tpend_linear + T_arm_linear
T = (1/2) * (Jeq*(theta_dot**2)) + (1/2) * (Jp*(alpha_dot**2)) + (1/2) * (Mp*vx**2) + (1/2) * (Mp*vy**2)

# Lagrangian
L = T - V

# Finding the Lagrangian's derivatives (for theta)
LE1 = sp.diff(sp.diff(L, theta_dot), t) - sp.diff(L, theta)

# Finding the Lagrangian's derivatives (for alpha)
LE2 = sp.diff(sp.diff(L, alpha_dot), t) - sp.diff(L, alpha)

# Solutions
solutions = sp.solve(
  [LE1, LE2],
  [d2_theta, d2_alpha]
)

sol_theta = sp.simplify(solutions[d2_theta])
sol_alpha = sp.simplify(solutions[d2_alpha])

E1 = sol_theta - tal + Beq*theta_dot
E2 = sol_alpha + Bp*alpha_dot

# Auxiliar Variables
a, b, c, d, e = sp.symbols('a b c d e')

_a = Mp*lp**2
_b = Mp*lp*r
_c = Jeq*Jp
_d = Mp*g*lp

simplified_e1 = E1.subs({
  _a: a,
  _b: b,
  _c: c,
  _d: d,
})

simplified_e2 = E2.subs({
  _a: a,
  _b: b,
  _c: c,
  _d: d,
})

sp.print_latex(
  sp.simplify(simplified_e1),
)

print(' ')

sp.print_latex(
  sp.simplify(simplified_e2),
)

print(' ')



