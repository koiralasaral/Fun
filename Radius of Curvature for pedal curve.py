import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the symbol and the tangential (pedal) function
theta = sp.symbols('theta', real=True)
p_expr = 1 + 0.5 * sp.cos(theta)  # Pedal equation: p(θ) = 1 + 0.5*cos(θ)
dp_expr = sp.diff(p_expr, theta)
d2p_expr = sp.diff(p_expr, theta, 2)

# Radius of curvature for a tangential polar curve:
# R(θ) = [p^2 + (dp/dθ)^2]^(3/2) / | p^2 + 2(dp/dθ)^2 - p*(d²p/dθ²) |
R_expr = (p_expr**2 + dp_expr**2)**(sp.Rational(3, 2)) / sp.Abs(p_expr**2 + 2*dp_expr**2 - p_expr * d2p_expr)

# Convert to a numerical function and evaluate over a range of θ
R_func = sp.lambdify(theta, R_expr, 'numpy')
theta_vals = np.linspace(0, 2 * np.pi, 400)
R_vals = R_func(theta_vals)

# Plot the radius of curvature versus θ
plt.figure(figsize=(8, 5))
plt.plot(theta_vals, R_vals, label='Radius of Curvature R(θ)')
plt.xlabel('θ (radians)')
plt.ylabel('Radius of Curvature R')
plt.title('Radius of Curvature for the Pedal Curve p = 1 + 0.5 cos(θ)')
plt.legend()
plt.grid(True)
plt.show()
