import numpy as np
import matplotlib.pyplot as plt

sampleDim = 2
sampleSize = 10000

# Sigma = I

sample = np.random.normal(size=(sampleSize,sampleDim))

plt.plot(sample[:,0],sample[:,1],'b.')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

# Sigma = [[1.0,.5],[.5,1.0]]

A = [[np.sqrt(3)*.5,np.sqrt(2)*.5],[np.sqrt(3)*.5,-np.sqrt(2)*.5]]
sample = np.random.normal(size=(sampleSize,sampleDim))

for n in range(sampleSize):
	sample[n] = np.dot(A,sample[n])

plt.plot(sample[:,0],sample[:,1],'r.')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
