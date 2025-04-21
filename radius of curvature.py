import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Explicit Function Example: f(x) = sin(x) -----------------------
# Define the symbol and function
x = sp.symbols('x', real=True)
f_expr = sp.sin(x)

# Compute first and second derivatives symbolically
fprime_expr = sp.diff(f_expr, x)
fprime2_expr = sp.diff(f_expr, x, 2)

# Define the curvature expression for y=f(x):
# κ(x) = |f''(x)| / (1 + (f'(x))^2)^(3/2)
curvature_expr = sp.Abs(fprime2_expr) / (1 + fprime_expr**2)**(sp.Rational(3, 2))
radius_expr = 1 / curvature_expr  # radius of curvature

# Display the symbolic expressions
print("f(x) =", f_expr)
print("f'(x) =", fprime_expr)
print("f''(x) =", fprime2_expr)
print("Curvature κ(x) =", sp.simplify(curvature_expr))
print("Radius of Curvature R(x) =", sp.simplify(radius_expr))

# Create numerical functions from symbolic expressions using lambdify
f = sp.lambdify(x, f_expr, 'numpy')
fprime = sp.lambdify(x, fprime_expr, 'numpy')
curvature = sp.lambdify(x, curvature_expr, 'numpy')
radius_func = sp.lambdify(x, radius_expr, 'numpy')

# Create a range of x values to evaluate the function and curvature
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_vals = f(x_vals)
k_vals = curvature(x_vals)
R_vals = radius_func(x_vals)

# ------------------ Osculating Circle Visualization at a Chosen Point ------------------
# Choose a point (x0, f(x0)); here we select x0 = 0 for demonstration.
x0 = 0.0
y0 = f(x0)
k0 = curvature(x0)
R0 = 1 / k0 if k0 != 0 else np.inf  # radius of osculating circle

# Compute the derivative at the point to define the tangent direction.
fprime0 = fprime(x0)

# For a function, the unit tangent vector is:
#    T = (1, f'(x0)) / sqrt(1+f'(x0)^2)
# and a candidate (unnormalized) normal vector is:
#    N = (-f'(x0), 1)
T0 = np.array([1, fprime0])
T0_unit = T0 / np.linalg.norm(T0)
N0 = np.array([-fprime0, 1])
N0_unit = N0 / np.linalg.norm(N0)

# The osculating circle center is located at 
# (x0, f(x0)) plus R0 in the direction of the unit normal.
# We adjust the sign based on f''(x0) (i.e. the concavity).
f2_x0 = float(fprime2_expr.subs(x, x0))
if f2_x0 >= 0:
    center = np.array([x0, y0]) + R0 * N0_unit
else:
    center = np.array([x0, y0]) - R0 * N0_unit

# Compute points on the osculating circle
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = center[0] + R0 * np.cos(theta)
circle_y = center[1] + R0 * np.sin(theta)

# --------------------------- Plotting the Function and Osculating Circle ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = sin(x)')
plt.plot(x0, y0, 'ro', label=f'Point of osculation (x={x0})')
plt.plot(circle_x, circle_y, 'g--', label=f'Osculating circle (R = {R0:.2f})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function f(x)=sin(x) and Its Osculating Circle at x=0')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# --------------------------- Plotting Curvature and Radius of Curvature ---------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('x')
ax1.set_ylabel('Curvature κ(x)', color='tab:red')
ax1.plot(x_vals, k_vals, color='tab:red', label='Curvature κ(x)')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('Radius of Curvature R(x)', color='tab:blue')
ax2.plot(x_vals, R_vals, color='tab:blue', label='Radius of Curvature R(x)')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

plt.title('Curvature and Radius of Curvature for f(x)=sin(x)')
plt.grid(True)
plt.show()
# ----------------------- Parametric Curve Example: Cycloid -----------------------
# Define the parameter
t = sp.symbols('t', real=True)
x_expr = t - sp.sin(t)
y_expr = 1 - sp.cos(t)

# First and second derivatives for x(t) and y(t)
xprime_expr = sp.diff(x_expr, t)
xprime2_expr = sp.diff(x_expr, t, 2)
yprime_expr = sp.diff(y_expr, t)
yprime2_expr = sp.diff(y_expr, t, 2)

# Define the curvature for a parametric curve:
# κ(t) = |x'(t) * y″(t) - y'(t) * x″(t)| / ( (x'(t)^2 + y'(t)^2)^(3/2) )
curvature_param_expr = sp.Abs(xprime_expr * yprime2_expr - yprime_expr * xprime2_expr) / ( (xprime_expr**2 + yprime_expr**2)**(sp.Rational(3, 2)) )
radius_param_expr = 1 / curvature_param_expr

print("\nParametric Cycloid:")
print("x(t) =", x_expr)
print("y(t) =", y_expr)
print("Curvature κ(t) =", sp.simplify(curvature_param_expr))
print("Radius of Curvature R(t) =", sp.simplify(radius_param_expr))

# Lambdify the expressions for numeric evaluation
x_func = sp.lambdify(t, x_expr, 'numpy')
y_func = sp.lambdify(t, y_expr, 'numpy')
curvature_param = sp.lambdify(t, curvature_param_expr, 'numpy')
radius_param = sp.lambdify(t, radius_param_expr, 'numpy')

# Create a range of parameter values.
t_vals = np.linspace(0, 4 * np.pi, 400)
x_vals_param = x_func(t_vals)
y_vals_param = y_func(t_vals)
k_vals_param = curvature_param(t_vals)
R_vals_param = radius_param(t_vals)

# Plot the cycloid
plt.figure(figsize=(10, 6))
plt.plot(x_vals_param, y_vals_param, label='Cycloid: x=t-sin(t), y=1-cos(t)')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Cycloid Curve')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plot curvature and radius for the cycloid
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('t')
ax1.set_ylabel('Curvature κ(t)', color='tab:red')
ax1.plot(t_vals, k_vals_param, color='tab:red', label='Curvature κ(t)')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('Radius of Curvature R(t)', color='tab:blue')
ax2.plot(t_vals, R_vals_param, color='tab:blue', label='Radius of Curvature R(t)')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

plt.title('Curvature and Radius of Curvature for the Cycloid')
plt.grid(True)
plt.show()
