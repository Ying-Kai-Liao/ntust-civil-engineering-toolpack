import matplotlib.pyplot as plt
import numpy as np

from sympy import lambdify, symbols, cos, sin, sqrt

# Define the symbols
Fb = symbols('Fb')

Mr = (2*30/Fb)*11*9.5 + 1089*34/3
Mr_simplified = Mr.simplify()
Mr_function = lambdify(Fb, Mr_simplified, 'numpy')

Md = (83-2*20/Fb)*4*2 + 136*8/3 + (151-2*30/Fb)*11*9.5 + 1089*34/3
Md_simplified = Md.simplify()
Md_function = lambdify(Fb, Md_simplified, 'numpy')

print(Mr_function(1))
print(Md_function(1))


Fb_values = np.linspace(0.4, 1, 800)

PpLp_values = Mr_function(Fb_values)
PaLa_values = Md_function(Fb_values)

Final_values = PpLp_values / PaLa_values

index_closest_to_one = np.argmin(np.abs(Final_values - 1))
Fb_estimation = Fb_values[index_closest_to_one]

print("Fb is roughly about: ", Fb_estimation)

plt.figure(figsize=(10, 5))
plt.plot(Fb_values, Final_values, label='PpLp/PaLa(Fb)')
plt.title('Plot Mr/Md in respect of Fb')
plt.xlabel('Fb')
plt.ylabel('Mr/Md')
plt.legend()
plt.grid(True)
plt.show()
