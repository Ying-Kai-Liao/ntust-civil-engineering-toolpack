

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer, make_pipeline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = [
    [0, 9872],
    [0.1, 9740],
    [0.25, 9728],
    [0.5, 9719],
    [1, 9709],
    [2.25, 9694],
    [4, 9682],
    [6.25, 9672],
    [9, 9664],
    [12.25, 9657],
    [16, 9653],
    [20.25, 9647],
    [25, 9644],
    [30.25, 9641],
    [36, 9639],
    [42.25, 9637],
    [49, 9635],
    [56.25, 9623],
    [64, 9622],
    [72.25, 9622],
    [81, 9621],
    [90.25, 9620],
    [100, 9620],
    [110.25, 9619],
    [121, 9618],
    [132.25, 9618],
    [144, 9617],
    [1440, 9601]
]

def combined_decay(x, a, b, c, d, e):
    result = np.array([])
    for i in x:
        value = a * np.exp(-b * i) + c * i**d + e
        result = np.append(result, value)
    return result

# Creating a DataFrame
df = pd.DataFrame(data, columns=['Time', 'Value'])

X = df['Time'].values.reshape(-1, 1)
y = df['Value'].values

# 使用sqrt資料來回歸，再使用原始資料呈現
X_sqrt = np.sqrt(X)
# X_log = X

X_subset = X_sqrt[2:9]  # Python uses 0-based indexing
y_subset = y[2:9]


# Setup and fit the model
polyreg = make_pipeline(PolynomialFeatures(2), LinearRegression())
polyreg.fit(X_sqrt[:-1], y[:-1])

popt, pcov = curve_fit(combined_decay, X_sqrt, y, p0=(50, 0.1, -150, 0.05, 9800), maxfev=10000)

polyreg1 = make_pipeline(LinearRegression())
polyreg1.fit(X_subset, y_subset)


coefficients = [polyreg.named_steps['linearregression'].intercept_] + polyreg.named_steps['linearregression'].coef_[-3:].tolist()
coefficients1 = [polyreg1.named_steps['linearregression'].intercept_] + polyreg1.named_steps['linearregression'].coef_.tolist()

p = np.polynomial.Polynomial(coefficients)
p1 = np.polynomial.Polynomial(coefficients1)

# Generate points for plotting from the minimum to maximum log-transformed X values
# 需要大量點來描繪開根號情形
X_plot = np.linspace(X.min(), X.max(), 10000).reshape(-1, 1)
X_plot_sqrt = np.sqrt(X_plot)
print(X_plot_sqrt)
y_pred = p(X_plot_sqrt)
y_pred_curve_fit = combined_decay(X_plot_sqrt, *popt)

X_plot_subset = np.linspace(X_subset.min(), X[:-10].max(), 100).reshape(-1, 1)
y_pred_subset = polyreg1.predict(np.sqrt(X_plot_subset))

print(f'Fitted parameters: a={popt[0]}, b={popt[1]}, c={popt[2]}, d={popt[3]}, e={popt[4]}')

def forward(x):
    return (x)**(1/2)

def inverse(x):
    return x**2

# Plot the results
plt.figure(figsize=(8, 6))
plt.xscale('function', functions=(forward, inverse))
# plt.scatter(X_log, y, color='gray')
plt.scatter(X, y, color='gray')

# plt.plot(X**(1/2), func(X.tolist(), *popt), 'r-', label="Fitted Curve")

plt.plot(X_plot, y_pred_curve_fit, color='green', label='Exponential Decay Fit')
plt.plot(X_plot_subset, y_pred_subset, color='red', label='Fitted model')

plt.xlabel('Log(Time)')
plt.ylabel('Value')
plt.title('Linear Regression with Log-Transformed X')
# plt.xticks(xi, X)
plt.legend()
plt.xlim(0, 1500)  # X-axis range from 0 to 5
# plt.ylim(9600, 10000) # Y-axis range from 0 to 20
plt.show()

# 計算壓密係數 C_v
h = 9.936
# print ("t50: ", 10**t_50[0])
# print ("C_v: ", 0.197 * h**2 / 10**t_50[0] / 100 / 60)
