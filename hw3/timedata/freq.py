import numpy as np
import matplotlib.pyplot as plt
import heapq as he
from data import *

small,med,large = load_data()
dim_str = ['small','med','large']
dim = [eval(name).size for name in dim_str]

# Ex. a

for n in range(len(dim)):
	f = eval(dim_str[n])
	ff = np.fft.fft(f)
	fr = np.fft.fftfreq(dim[n],d=1./dim[n])

	for i in range(dim[n]):
		plt.plot([fr[i],fr[i]],[0.,np.abs(ff[i])],'b')
	# save figure to file
	plt.show()

# Ex. b

a = np.abs(np.fft.ifft(large))
a_index = a.argsort()[-3:][::-1]
a_max = [a[i] for i in list(a_index)]
print a_max
print a_index
