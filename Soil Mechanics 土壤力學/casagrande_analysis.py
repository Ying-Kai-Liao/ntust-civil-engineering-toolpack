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
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['axes.unicode_minus'] = False

def soil_consolidation_results(data, h, plot_number):
    """
    This function generates soil consolidation results using Casagrande's method.
    It involves fitting a polynomial regression model to log-transformed time data and corresponding micrometer readings.

    Parameters:
    data (list of lists): The input data with time and micrometer readings.
    h (float): The height of the soil sample.
    plot_number (int): The plot number for naming the output file.

    Returns:
    None
    """

    # Create a DataFrame with the input data
    df = pd.DataFrame(data, columns=['Time', 'Value'])

    X = df['Time'].values.reshape(-1, 1)
    X_calc = X[:-1]
    y = df['Value'].values / 1000
    print(y[0])
    y_calc = y[:-1]

    # Log transform the time data
    X_log = np.log10(X_calc)

    # Subsets of the data for specific polynomial fits
    X_subset = X_log[4:9]
    y_subset = y[4:9]

    X_subset2 = X_log[-10:-2]
    y_subset2 = y[-10:-2]

    # Fit polynomial regression models
    polyreg = make_pipeline(PolynomialFeatures(3), LinearRegression())
    polyreg.fit(X_log, y_calc)

    polyreg1 = make_pipeline(LinearRegression())
    polyreg1.fit(X_subset, y_subset)

    polyreg2 = make_pipeline(LinearRegression())
    polyreg2.fit(X_subset2, y_subset2)

    # Generate points for plotting
    
    X_plot = np.linspace(X_log.min(), X_log.max(), 1000).reshape(-1, 1)
    y_pred = polyreg.predict(X_plot)

    X_plot_subset = np.linspace(X_subset.min(), X_log[:-2].max(), 100).reshape(-1, 1)
    y_pred_subset = polyreg1.predict(X_plot_subset)

    X_plot_subset2 = np.linspace(X_subset.min(), X_subset2.max(), 100).reshape(-1, 1)
    y_pred_subset2 = polyreg2.predict(X_plot_subset2)

    # Polynomial coefficients
    coefficients = [polyreg.named_steps['linearregression'].intercept_] + polyreg.named_steps['linearregression'].coef_[-3:].tolist()
    coefficients1 = [polyreg1.named_steps['linearregression'].intercept_] + polyreg1.named_steps['linearregression'].coef_.tolist()
    coefficients2 = [polyreg2.named_steps['linearregression'].intercept_] + polyreg2.named_steps['linearregression'].coef_.tolist()

    p = np.polynomial.Polynomial(coefficients)
    p1 = np.polynomial.Polynomial(coefficients1)
    p2 = np.polynomial.Polynomial(coefficients2)

    X_plot_t = np.linspace(X.min(), X[-2], 10000).reshape(-1, 1)
    y_pred_t = p(np.log10(X_plot_t))

    s_100_x = (p1 - p2).roots()
    s_100 = p1(s_100_x[0])

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.scatter(X, y, color='gray')
    plt.plot(X_plot_t, y_pred_t, color='gray')
    plt.plot(10 ** X_plot_subset, y_pred_subset, color='green')
    plt.plot(10 ** X_plot_subset2, y_pred_subset2, color='blue')

    t1 = 0.2
    s1 = polyreg.predict(np.array([[np.log10(t1)]]))[0]
    plt.axvline(x=t1, color='orange', linestyle='--', label=f't1={t1}')
    plt.axhline(y=s1, color='orange', linestyle='-.', label=f's1={s1}')

    t2 = 0.8
    s2 = polyreg.predict(np.array([[np.log10(t2)]]))[0]
    plt.axvline(x=t2, color='orange', linestyle='--', label=f't2={t2}')
    plt.axhline(y=s2, color='orange', linestyle='-.', label=f's2={s2}')

    s_0 = s1 + s1 - s2
    plt.axhline(y=s_0, color='orange', linestyle='-.', label=f's0={s_0}')
    plt.plot(10 ** s_100_x, s_100, "ro")
    plt.axhline(y=s_100, color='orange', linestyle='-.', label=f's100={s_100}')

    s_50 = (s_0 + s_100) / 2
    plt.axhline(y=s_50, color='orange', linestyle='-.', label=f's50={s_50}')

    t_50 = (p1 - s_50).roots()
    plt.axvline(x=10 ** t_50[0], color='orange', linestyle='--', label=f't50={10**t_50[0]}')

    plt.xlabel('時間(min)')
    plt.ylabel('沈陷讀數(mm)')
    plt.title(f'Casagrande CV{plot_number}')
    plt.legend()

    # Save the plot
    filename = f"plot_casagrande_{plot_number}.png"
    plt.savefig(filename)
    plt.close()

    # Calculate and print the consolidation coefficient (C_v)
    t_50_value = 10**t_50[0]
    C_v = 0.197 * h**2 / t_50_value / 100 / 60
    
    results = (t1, s1, t2, s2, s1-s2, s_0, s_100, s_50, t_50_value, h, C_v)

    # Save the results to a text file in a comma-separated format
    with open(f"results_c_{plot_number}.txt", "w", encoding='utf-8') as f:
        f.write(", ".join(map(str, results)) + "\n")
    with open(f"results_c.txt", "a", encoding='utf-8') as f:
        f.write(", ".join(map(str, results)) + "\n") 

    print(f"Plot saved as {filename}")
    print(f"Results saved as results_{plot_number}.txt")
    print(results)

# Data for each day
days_data = [
    ([
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
    ], 9.936),
    ([
        # [0, 9601],
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
        # [0, 9270],
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
        # [0, 9027],
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
        # [0, 8710],
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
        # [0, 8246],
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


