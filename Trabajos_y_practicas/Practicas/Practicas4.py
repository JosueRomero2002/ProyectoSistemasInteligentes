import sympy as sp

x = sp.symbols('x')

f = x**2 + 2*x + 1

print("Derivada de f(x) = x^2 + 2x + 1: " + str(sp.diff(f, x)))