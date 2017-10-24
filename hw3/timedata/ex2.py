import numpy as np
import matplotlib.pyplot as plt

f = np.loadtxt('data2049.txt')
n = f.shape[0]

ff = np.fft.fft(f)
fr = np.fft.fftfreq(n,d=1./n)

plt.plot(list(fr[(n-1)/2+1:])+list(fr[0:(n-1)/2]),list(np.abs(ff)[(n-1)/2+1:])+list(np.abs(ff)[0:(n-1)/2]))
plt.show()
