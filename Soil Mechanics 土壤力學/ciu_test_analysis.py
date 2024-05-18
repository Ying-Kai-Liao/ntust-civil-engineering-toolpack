"""
This code is used to generate the resulting outcomes of a CIU (Consolidated Isotropically Undrained) test.
It involves fitting a regression line to minimize the squared distance to a set of semi-circles, which represent the test data.
The process includes:

1. Importing necessary libraries for numerical computation and plotting.
2. Defining semi-circles with specified centers (representing load + maximum strain) and radii (maximum strain).
3. Calculating the distance from the regression line to each semi-circle and adjusting this distance by subtracting the radius.
4. Using an optimization algorithm to adjust the regression line parameters (slope and intercept) to minimize the squared adjusted distance.
5. Plotting the resulting optimized regression line along with the semi-circles.
6. Ensuring the plot has a 1:1 aspect ratio and includes the x and y axes for better visualization.

This implementation uses NumPy and SciPy for numerical computations and optimization, and Matplotlib for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 圓心(荷重 + 最大應變), 半徑(最大應變)
semi_circles = [{'center': (.9703, 0), 'radius': .4703},
                {'center': (2.18995, 0), 'radius': 1.18995},
                {'center': (3.19355, 0), 'radius': 1.19355},
                {'center': (4.6525, 0), 'radius': 1.6525},
                {'center': (5.19245, 0), 'radius': 1.19245},
                ]

# Initial regression line (y = mx + b)
m, b = 1, 0
def distance_square_to_semicircle(line_params, semi_circle):
    m, b = line_params
    center_x, center_y = semi_circle['center']
    radius = semi_circle['radius']
    # Distance from point to line formula
    distance = abs(m * center_x - center_y + b) / np.sqrt(m**2 + 1)
    # # Adjust distance by radius
    adjusted_distance = (distance - radius) ** 2
    print(f'Center: ({center_x}, {center_y}), Radius: {radius}, Distance: {adjusted_distance}')
    return adjusted_distance
def objective_function(line_params):
    total_distance = 0
    for semi_circle in semi_circles:
        total_distance += distance_square_to_semicircle(line_params, semi_circle)
    return total_distance
initial_params = [m, b]
result = minimize(objective_function, initial_params)
optimized_m, optimized_b = result.x
if optimized_m < 0:
    optimized_m = -optimized_m
    optimized_b = -optimized_b
print(f'Optimized line: y = {optimized_m}x + {optimized_b}')
x_vals = np.linspace(-3, 6, 100)
y_vals = optimized_m * x_vals + optimized_b

plt.figure()
plt.plot(x_vals, y_vals, label='Optimized Regression Line')

for sc in semi_circles:
    circle = plt.Circle(sc['center'], sc['radius'], color='r', fill=False)
    plt.gca().add_patch(circle)

plt.xlim(0, 10)
plt.ylim(0, 6)
plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis
plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()
