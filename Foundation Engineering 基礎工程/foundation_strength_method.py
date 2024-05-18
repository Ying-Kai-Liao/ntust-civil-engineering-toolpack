import matplotlib.pyplot as plt
import numpy as np

from sympy import lambdify, symbols, cos, sin, sqrt

# Define the symbols
phi, delta, beta, theta = symbols('phi delta beta theta')

# Given Kp expression
Kp = cos(phi + theta)**2 / (cos(theta)**2 * cos(delta - theta) * (1 - sqrt(sin(phi + delta)*sin(phi + beta) / (cos(delta - theta)*cos(beta - theta))))**2)

Kp_substituted = Kp.subs({delta: (2/3)*phi, beta: 0, theta: 0})

# Simplify the expression after substitution
Kp_simplified = Kp_substituted.simplify()

Kp_function = lambdify(phi, Kp_simplified, 'numpy')

print(Kp_function(30/180*np.pi))

Ka = cos(phi - theta)**2 / (cos(theta)**2 * cos(delta + theta) * (1 + sqrt(sin(phi + delta)*sin(phi - beta) / (cos(delta + theta)*cos(theta - beta))))**2)

Ka_substituted = Ka.subs({delta: (2/3)*phi, beta: 0, theta: 0})

# Simplify the expression after substitution
Ka_simplified = Ka_substituted.simplify()

Ka_function = lambdify(phi, Ka_simplified, 'numpy')

print(Ka_function(30/180*np.pi))


phi_values = np.linspace(0.1, 0.5, 800)

Fb_values = (0.577/np.tan(phi_values))

# Evaluate the Kp function for each value of phi
Ka_values = Ka_function(phi_values)
Kp_values = Kp_function(phi_values)

Kah_values=np.cos(20/180*np.pi) * Ka_values
Kph_values=np.cos(20/180*np.pi) * Kp_values

PaLa_values = 3427.75 * Kah_values / 0.279
PpLp_values = 16514.57 * Kph_values / 5.737

Final_values = PpLp_values / PaLa_values

index_closest_to_one = np.argmin(np.abs(Final_values - 1))
Fb_estimation = Fb_values[index_closest_to_one]

print("Fb is roughly about: ", Fb_estimation)

# Plot the function using matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))  # 1 row, 2 columns

# Plot on the first subplot
ax1.plot(Fb_values, Final_values, label='PpLp/PaLa(Fb)')
ax1.set_title('Plot PpLp/PaLa in respect of Fb')
ax1.set_xlabel('Fb')
ax1.set_ylabel('PpLp/PaLa')
ax1.legend()
ax1.grid(True)

# Plot on the second subplot
ax2.plot(phi_values, Fb_values, label='Fb(phi)')
ax2.set_title('Plot of Fb in respect of phi')
ax2.set_xlabel('phi')
ax2.set_ylabel('Fb')
ax2.legend()
ax2.grid(True)

plt.show()