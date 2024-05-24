import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer, make_pipeline
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['axes.unicode_minus'] = False

data = [
    [0.000, 0.6197],
    [0.125, 0.5874],
    [0.250, 0.5762],
    [0.500, 0.5606],
    [1.000, 0.5409],
    [2.000, 0.5152],
    [4.000, 0.4844],
    [2.000, 0.4844],
    [1.000, 0.4858],
    [0.500, 0.4893],
    [0.250, 0.4947],
    [0.125, 0.5005],
    [0.000, 0.4993],
    [0.125, 0.5197],
    [0.250, 0.5184],
    [0.500, 0.5148],
    [1.000, 0.5091],
    [2.000, 0.5012],
    [4.000, 0.4921],
    [8.000, 0.4345]
]

df = pd.DataFrame(data, columns=['sigma', 'e'])

X = df['sigma'].values.reshape(-1, 1)
y = df['e'].values

X_log = np.log10(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_log, y, color='gray')

plt.xlabel('logσ (kg/cm^2)')
plt.ylabel('e')

plt.legend()
plt.title('利用 e-log σ 曲線圖(無解壓再壓段)計算σ’c')
plt.show()

x1, y1 = [-0.002, 0.5404]
x2, y2 = [0.0606, 0.4911]

x3, y3 = [-0.907, 0.5202]
x4, y4 = [0.602, 0.4836]

slope1 = -(y2 - y1) / (x2 - x1)
intercept1 = y1 - slope1 * x1

slope2 = -(y4 - y3) / (x4 - x3)
intercept1 = y3 - slope2 * x3

print ("Cc = ", slope1)
print ("Cs = ", slope2)