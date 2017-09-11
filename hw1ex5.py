import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1.,1.,2001)
f = np.cos(np.pi * x * .5)
p = 15 * (np.pi**2 - 12) / np.pi**3 * x**2 + 3 / np.pi**3 * (20 -np.pi**2)
t = - np.pi**2 / 8 * x**2 + 1

plt.plot(x,f,label='f')
plt.plot(x,p,label='P_2')
plt.plot(x,t,label='T_2')
plt.legend()
plt.show()
