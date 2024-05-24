import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer, make_pipeline
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['axes.unicode_minus'] = False

data = [
    # [0.000, 0.6197],
    [0.125, 0.5874],
    [0.250, 0.5762],
    [0.500, 0.5606],
    [1.000, 0.5409],
    [2.000, 0.5152],
    [4.000, 0.4844],
    [8.000, 0.4345]
]


df = pd.DataFrame(data, columns=['sigma', 'e'])

X = df['sigma'].values.reshape(-1, 1)
y = df['e'].values

X_log = np.log10(X)

spline = UnivariateSpline(X_log, y, s=0.1)

X_plot = np.linspace(X_log.min(), X_log.max(), 1000)
X_plot_tangent = np.linspace(X_log.min(), X_log.max() + 2, 1000)
y_spline = spline(X_plot)

first_derivative = spline.derivative(n=1)
second_derivative = spline.derivative(n=2)

curvature = np.abs(second_derivative(X_plot[100:-300])) / (1 + first_derivative(X_plot[100:-300])**2)**1.5

max_curvature_index = np.argmax(curvature)
max_curvature_point = X_plot[100:-300][max_curvature_index]
max_curvature_value = spline(max_curvature_point)

tangent_slope = first_derivative(max_curvature_point)
tangent_intercept = spline(max_curvature_point) - tangent_slope * max_curvature_point

def tangent_line(x):
    return tangent_slope * x + tangent_intercept

# Calculate the angle of the tangent line
tangent_angle = np.arctan(tangent_slope)

# Calculate the angle of the bisector (halfway between the tangent and the x-axis)
bisector_angle = tangent_angle / 2

# Define the bisector line function
def bisector_line(x, x0, y0, angle):
    return y0 + np.tan(angle) * (x - x0)

X_plot_bisector = X_plot_tangent
y_bisector = bisector_line(X_plot_bisector, max_curvature_point, max_curvature_value, bisector_angle)

x1 = X_log[-2]
x2 = X_log[-1]
y1 = y[-2]
y2 = y[-1]
print(x1, x2, y1, y2)
slope = (y2 - y1) / (x2 - x1)
intercept = y1 - slope * x1
def line(x):
    return slope * x + intercept

X_plot_line = X_plot_tangent
y_line = line(X_plot_line)

def find_intersection(f1, f2, x0):
    from scipy.optimize import fsolve
    return fsolve(lambda x: f1(x) - f2(x), x0)

# Use an initial guess for the intersection point
initial_guess = (x1 + x2) / 2
intersection_x = find_intersection(lambda x: bisector_line(x, max_curvature_point, max_curvature_value, bisector_angle), line, initial_guess)[0]
intersection_y = line(intersection_x)

plt.figure(figsize=(8, 6))
plt.scatter(X_log, y, color='gray')
plt.plot(X_plot, y_spline, color='red')
plt.plot(X_plot_tangent, tangent_line(X_plot_tangent), color='green', linestyle='--', label='Tangent line')
plt.scatter(max_curvature_point, spline(max_curvature_point), color='orange', label='Max curvature point')
plt.plot(X_plot_bisector, y_bisector, color='blue', linestyle='-.', label='Bisector line')
plt.axhline(y=max_curvature_value, linestyle='--', color='green')
plt.plot(X_plot_line, y_line, color="orange")
plt.scatter(intersection_x, intersection_y, color='black', label=f'預壓密應力σ′c = {10**intersection_x}')
plt.xlabel('logσ (kg/cm^2)')
plt.ylabel('e')
plt.legend()
plt.title('利用 e-log σ 曲線圖(無解壓再壓段)計算σ’c')
plt.show()
