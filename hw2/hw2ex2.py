import numpy as np
import matplotlib.pyplot as plt

sampleDim = 100
sampleSize = 10000

# (a)

sample = np.random.normal(size=(sampleSize,sampleDim))
sampleNormSq = np.sum(np.square(sample),axis=1)

plt.hist(sampleNormSq)
plt.show()

p = np.sum([((x >= 80) and (x <= 120)) for x in sampleNormSq],dtype=np.float32) / sampleSize

print p

# (b)

sample = np.random.normal(size=(sampleSize,sampleDim))
sampleNorm = np.linalg.norm(sample,axis=1)

plt.hist(sampleNorm)
plt.show()

p = np.sum([((x >= 9) and (x <= 11)) for x in sampleNorm],dtype=np.float32) / sampleSize

print p

# (c)

Sigma = .5 * np.ones((sampleDim,sampleDim)) + np.diag(.5 * np.ones((sampleDim)))
U, S, V = np.linalg.svd(Sigma)
A = np.dot(U,np.diag(np.sqrt(S)))

sample = np.random.normal(size=(sampleSize,sampleDim)) 
for n in range(sampleSize):
	sample[n] = np.dot(A,sample[n])
sampleNormSq = np.sum(np.square(sample),axis=1)

plt.hist(sampleNormSq)
plt.show()

