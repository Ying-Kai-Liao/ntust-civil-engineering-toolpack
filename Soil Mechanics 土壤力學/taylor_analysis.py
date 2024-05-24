import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['axes.unicode_minus'] = False

def soil_consolidation_results(data, h, plot_number):

    # Creating a DataFrame
    df = pd.DataFrame(data, columns=['Time', 'Value'])

    # Extracting X and y
    X = df['Time'].values.reshape(-1, 1)
    y = df['Value'].values / 1000

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
    popt, pcov = curve_fit(combined_decay, X_sqrt, y, p0=(50, 0.1, -150, 0.05, 9.800), maxfev=20000)


    polyreg1 = make_pipeline(LinearRegression())
    polyreg1.fit(X_subset, y_subset)

    # Extract the slope (coefficient) and intercept
    slope = polyreg1.named_steps['linearregression'].coef_[0]
    intercept = polyreg1.named_steps['linearregression'].intercept_
    
    S0 = intercept
    A = (y[-1] - intercept) / slope
    # Adjust the slope
    adjusted_slope = slope / 1.15
    A_new = (y[-1] - intercept) / adjusted_slope

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
    sqrt_t_90 = intersection_x = (x_range[min_index])
    S90 = intersection_y = combined_decay_calc(intersection_x, *popt)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sqrt, y, color='blue', label='Data')
    plt.plot(X_plot, y_fit, color='red', label='Combined Decay Fit')
    plt.plot(X_plot_subset, y_pred_subset, color='orange', label='Fitted model')
    plt.plot(X_plot_subset, y_pred_adjusted, color='orange', label='Adjusted Linear Fit')
    plt.axvline(x=sqrt_t_90, color='orange', linestyle='-.', label=f'sqrt(t90)')
    plt.xlabel('時間(min^1/2)')
    plt.ylabel('沈陷讀數(mm)')
    plt.title(f'Taylor CV{plot_number}')
    plt.legend()
    
        # Save the plot
    filename = f"plot_taylor_{plot_number}.png"
    plt.savefig(filename)
    plt.close()
    
    t_90 = sqrt_t_90 ** 2
    Cv = 0.848 * h**2 / sqrt_t_90[0] ** 2 / 100 / 60
    
    results = (S0, A, A_new, S90[0], t_90[0], h, Cv)
    
    with open(f"results_t_{plot_number}.txt", "w", encoding='utf-8') as f:
        f.write(", ".join(map(str, results)) + "\n") 
    with open(f"results_t.txt", "a", encoding='utf-8') as f:
        f.write(", ".join(map(str, results)) + "\n") 
    
    print(f"Plot saved as {filename}")
    print(f"Results saved as results_{plot_number}.txt")
    print(results)

# Data for each day
days_data = [
    ([
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
    ], 9.936),
    ([
        [0, 9601],
        [0.1, 9552],
        [0.25, 9549],
        [0.5, 9545],
        [1, 9540],
        [2.25, 9531],
        [4, 9524],
        [6.25, 9517],
        [9, 9512],
        [12.25, 9506],
        [16, 9503],
        [20.25, 9501],
        [25, 9499],
        [30.25, 9497],
        [36, 9495],
        [42.25, 9494],
        [49, 9492],
        [56.25, 9491],
        [64, 9490],
        [72.25, 9489],
        [81, 9489],
        [90.25, 9488],
        [100, 9487],
        [110.25, 9486],
        [121, 9486],
        [132.25, 9485],
        [144, 9484],
        [1440, 9463]
    ], 9.8005),
    ([
        # [0, 9463],
        [0.1, 9401],
        [0.25, 9392],
        [0.5, 9385],
        [1, 9376],
        [2.25, 9363],
        [4, 9350],
        [6.25, 9342],
        [9, 9336],
        [12.25, 9331],
        [16, 9327],
        [20.25, 9324],
        [25, 9310],
        [30.25, 9308],
        [36, 9305],
        [42.25, 9304],
        [49, 9303],
        [56.25, 9301],
        [64, 9300],
        [72.25, 9299],
        [81, 9298],
        [90.25, 9297],
        [100, 9296],
        [110.25, 9295],
        [121, 9294],
        [132.25, 9294],
        [144, 9293],
        [1440, 9270]
    ], 9.7315),
    ([
        [0, 9270],
        [0.1, 9182],
        [0.25, 9172],
        [0.5, 9161],
        [1, 9149],
        [2.25, 9134],
        [4, 9123],
        [6.25, 9116],
        [9, 9110],
        [12.25, 9106],
        [16, 9102],
        [20.25, 9100],
        [25, 9097],
        [30.25, 9095],
        [36, 9093],
        [42.25, 9092],
        [49, 9090],
        [56.25, 9089],
        [64, 9088],
        [72.25, 9087],
        [81, 9086],
        [90.25, 9085],
        [100, 9084],
        [110.25, 9083],
        [121, 9082],
        [132.25, 9081],
        [144, 9080],
        [1440, 9027]
    ], 9.635),
    ([
        [0, 9027],
        [0.1, 8992],
        [0.25, 8979],
        [0.5, 8966],
        [1, 8949],
        [2.25, 8926],
        [4, 8908],
        [6.25, 8895],
        [9, 8886],
        [12.25, 8879],
        [16, 8868],
        [20.25, 8866],
        [25, 8862],
        [30.25, 8860],
        [36, 8857],
        [42.25, 8856],
        [49, 8853],
        [56.25, 8851],
        [64, 8850],
        [72.25, 8848],
        [81, 8846],
        [90.25, 8845],
        [100, 8844],
        [110.25, 8843],
        [121, 8842],
        [132.25, 8841],
        [144, 8840],
        [1440, 8710]
    ], 9.5135),
    ([
        [0, 8710],
        [0.1, 8515],
        [0.25, 8499],
        [0.5, 8483],
        [1, 8461],
        [2.25, 8434],
        [4, 8413],
        [6.25, 8403],
        [9, 8396],
        [12.25, 8388],
        [16, 8384],
        [20.25, 8382],
        [25, 8378],
        [30.25, 8376],
        [36, 8374],
        [42.25, 8372],
        [49, 8370],
        [56.25, 8368],
        [64, 8367],
        [72.25, 8365],
        [81, 8364],
        [90.25, 8362],
        [100, 8361],
        [110.25, 8360],
        [121, 8359],
        [132.25, 8358],
        [144, 8357],
        [1440, 8330]
    ], 9.355),
    ([
        [0, 8246],
        [0.1, 7998],
        [0.25, 7971],
        [0.5, 7937],
        [1, 7896],
        [2.25, 7849],
        [4, 7822],
        [6.25, 7808],
        [9, 7799],
        [12.25, 7792],
        [16, 7787],
        [20.25, 7783],
        [25, 7779],
        [30.25, 7776],
        [36, 7774],
        [42.25, 7772],
        [49, 7770],
        [56.25, 7768],
        [64, 7765],
        [72.25, 7764],
        [81, 7762],
        [90.25, 7760],
        [100, 7759],
        [110.25, 7757],
        [121, 7756],
        [132.25, 7755],
        [144, 7754],
        [1440, 7714]
    ], 9.123)
]

# Loop through each day's data and call the function
for i, (data, h) in enumerate(days_data):
    soil_consolidation_results(data, h, i + 1)


