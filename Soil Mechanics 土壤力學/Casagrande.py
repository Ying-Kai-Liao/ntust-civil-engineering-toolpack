"""
This code is used to generate 土壤壓密結果 (soil consolidation results) using Casagrande's method.
It involves fitting a polynomial regression model to log-transformed time data and corresponding 測微表讀數 (micrometer readings).
The process includes:

1. Importing necessary libraries for data manipulation, transformation, and visualization.
2. Creating a DataFrame with time (in minutes) and 測微表讀數 (micrometer readings) data points.
3. Applying a log transformation to the time data.
4. Fitting polynomial regression models to the entire dataset and specific subsets.
5. Generating predictions and plotting the results.
6. Highlighting key points and lines on the plot, such as fitted models for specific data subsets and intersections.
7. Calculating and printing the consolidation coefficient (C_v) based on the fitted model.

The plot is generated with a logarithmic x-axis to visualize the relationship between time and micrometer readings more effectively.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer, make_pipeline
import matplotlib.pyplot as plt

data = [
    # [0, 9872],
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

df = pd.DataFrame(data, columns=['Time', 'Value'])

X = df['Time'].values.reshape(-1, 1)
y = df['Value'].values

def log_transform(x):
    return np.log10(x)

X_log = np.log10(X)

X_subset = X_log[4:9]  
y_subset = y[4:9]

X_subset2 = X_log[-10:]  
y_subset2 = y[-10:]

polyreg = make_pipeline(PolynomialFeatures(3), LinearRegression())
polyreg.fit(X_log, y)

polyreg1 = make_pipeline(LinearRegression())
polyreg1.fit(X_subset, y_subset)

polyreg2 = make_pipeline(LinearRegression())
polyreg2.fit(X_subset2, y_subset2)

# Generate points for plotting from the minimum to maximum log-transformed X values
X_plot = np.linspace(X_log.min(), X_log.max(), 100).reshape(-1, 1)
y_pred = polyreg.predict(X_plot)

X_plot_subset = np.linspace(X_subset.min(), X_log[:-2].max(), 100).reshape(-1, 1)
y_pred_subset = polyreg1.predict(X_plot_subset)

X_plot_subset2 = np.linspace(X_subset.min(), X_subset2.max(), 100).reshape(-1, 1)
y_pred_subset2 = polyreg2.predict(X_plot_subset2)

coefficients = [polyreg.named_steps['linearregression'].intercept_] + polyreg.named_steps['linearregression'].coef_[-3:].tolist()
coefficients1 = [polyreg1.named_steps['linearregression'].intercept_] + polyreg1.named_steps['linearregression'].coef_.tolist()
coefficients2 = [polyreg2.named_steps['linearregression'].intercept_] + polyreg2.named_steps['linearregression'].coef_.tolist()

p = np.polynomial.Polynomial(coefficients)
p1 = np.polynomial.Polynomial(coefficients1)
p2 = np.polynomial.Polynomial(coefficients2)

X_plot_t = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred_t = p(np.log10(X_plot_t))

s_100_x = (p1-p2).roots()
s_100 = p1(s_100_x[0])

# Plot the results
plt.figure(figsize=(8, 6))
plt.xscale('log')
# plt.scatter(X_log, y, color='gray')
plt.scatter(X, y, color='gray')

plt.plot(X_plot_t, y_pred_t, color='gray', label='Fitted model')
plt.plot(10 ** X_plot_subset, y_pred_subset, color='green', label='Fitted model (5th-9th Data)')
plt.plot(10 ** X_plot_subset2, y_pred_subset2, color='blue', label='Fitted model (Last 10 Data)')

t1 = (0.2)
s1 = polyreg.predict(np.array([[np.log10(t1)]]))[0]
plt.axvline(x=t1, color='orange', linestyle='--', label = t1)
plt.axhline(y=s1, color='orange', linestyle='-.', label=f's1')

t2 = (0.8)
s2 = polyreg.predict(np.array([[np.log10(t2)]]))[0]
plt.axvline(x=t2, color='orange', linestyle='--', label = 't2') 
plt.axhline(y=s2, color='orange', linestyle='-.', label=f's2')

s_0 = s1 + s1 - s2
plt.axhline(y=s_0, color='orange', linestyle='-.', label=f's0')
plt.plot(10 ** s_100_x, s_100, "ro")
plt.axhline(y=s_100, color='orange', linestyle='-.', label=f's_100')

s_50 = (s_0 + s_100) / 2
plt.axhline(y=s_50, color='orange', linestyle='-.', label=f's_50')

t_50 = (p1 - s_50).roots()
plt.axvline(x=10 ** (t_50[0]), color='orange', linestyle='--', label=f't_50')

plt.xlabel('Log(Time)')
plt.ylabel('Value')
plt.title('Linear Regression with Log-Transformed X')
plt.legend()
plt.show()

# 計算壓密係數 C_v
h = 9.936
print ("t50: ", 10**t_50[0])
print ("C_v: ", 0.197 * h**2 / 10**t_50[0] / 100 / 60)
