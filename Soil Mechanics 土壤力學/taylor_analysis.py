import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Provided data
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

# Creating a DataFrame
df = pd.DataFrame(data, columns=['Time', 'Value'])

# Extracting X and y
X = df['Time'].values.reshape(-1, 1)
y = df['Value'].values

# Transform X to X_sqrt
X_sqrt = np.sqrt(X)

X_subset = X_sqrt[2:9]  # Python uses 0-based indexing
y_subset = y[2:9]

# Define the combined exponential and power-law decay function
def combined_decay(x, a, b, c, d, e):
    result = np.array([])
    # print (x)
    for i in x:
        value = a * np.exp(-b * i) + c * i**d + e
        result = np.append(result, value)
    return result

def combined_decay_calc(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * x**d + e
# Fit the function to the data using X_sqrt with increased maxfev
popt, pcov = curve_fit(combined_decay, X_sqrt, y, p0=(50, 0.1, -150, 0.05, 9800), maxfev=10000)


polyreg1 = make_pipeline(LinearRegression())
polyreg1.fit(X_subset, y_subset)

# Extract the slope (coefficient) and intercept
slope = polyreg1.named_steps['linearregression'].coef_[0]
intercept = polyreg1.named_steps['linearregression'].intercept_

# Adjust the slope
adjusted_slope = slope / 1.15

# Create the new adjusted linear model
def adjusted_linear_model(x):
    return adjusted_slope * x + intercept

# Generate points for plotting
X_plot = np.linspace(X_sqrt.min(), X_sqrt.max(), 100)
y_fit = combined_decay(X_plot, *popt)

X_plot_subset = np.linspace(X_subset.min(), X_sqrt[:-10].max(), 100).reshape(-1, 1)
y_pred_subset = polyreg1.predict(X_plot_subset)

y_pred_adjusted = adjusted_linear_model(X_plot_subset)

# Calculate the difference
def difference(x):
    return adjusted_linear_model(x) - combined_decay_calc(x, *popt)

# Find the intersection by minimizing the difference
# Find the intersection by minimizing the difference
x_range = np.linspace(X_sqrt[6], X_sqrt[-2], 10000)
differences = np.abs(difference(x_range))
min_index = np.argmin(differences)
sqrt_t_90 = (x_range[min_index])
print(sqrt_t_90, adjusted_linear_model(sqrt_t_90), combined_decay_calc(sqrt_t_90, *popt), difference(sqrt_t_90))
# intersection_y = combined_decay_calc(intersection_x, *popt)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_sqrt, y, color='blue', label='Data')
plt.plot(X_plot, y_fit, color='red', label='Combined Decay Fit')
plt.plot(X_plot_subset, y_pred_subset, color='orange', label='Fitted model')
plt.plot(X_plot_subset, y_pred_adjusted, color='orange', label='Adjusted Linear Fit')
plt.axvline(x=sqrt_t_90, color='orange', linestyle='-.', label=f'sqrt(t90)')
plt.xlabel('Sqrt(Time)')
plt.ylabel('Value')
plt.title('Combined Exponential and Power-Law Decay Fit to Data (Sqrt)')
plt.legend()
plt.show()


# 計算壓密係數 C_v
h = 9.936
print ("t90: ", sqrt_t_90 ** 2)
print ("C_v: ", 0.848 * h**2 / sqrt_t_90 ** 2 / 100 / 60)
