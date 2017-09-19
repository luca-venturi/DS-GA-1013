"""
@author: Luca Venturi
"""

import numpy as np
from mnist_tools import *
from plot_tools import *
from nearest_neighbors import compute_nearest_neighbors
import matplotlib.pyplot as plt

# part (i)

datafile = "mnist_all.mat" #Change if you put the file in a different path
train = load_train_data(datafile)

dataSize = len(train[0][0])
nDigits = 10
trainSampleSize = 100
trainSize = nDigits * trainSampleSize
trainMatrix = np.zeros((trainSize,dataSize))
for i in range(trainSize):
	trainMatrix[i] = train[int(i/trainSampleSize)][i%trainSampleSize]

trainMean = np.mean(trainMatrix, axis=0)
trainMatrixCentered = np.zeros((trainSize,len(train[0][0])))
for i in range(dataSize):
	trainMatrixCentered[i] = trainMatrix[i] - trainMean

trainU, trainS, trainV = np.linalg.svd(trainMatrixCentered)

plt.plot(trainS)
plt.show()

# part (ii)

plot_image_grid(trainV[:10,:], '10 first singular vectors')

# part (iii)

k = 30 # it's the best for the below test
principalDirections = trainV[:k,:]

trainProjectedCoeff = np.zeros((nDigits,trainSampleSize,k))
for i in range(nDigits):
	for j in range(trainSampleSize):
		trainProjectedCoeff[i][j] = np.dot(train[i][j],principalDirections.transpose())

test,testLabels = load_test_data(datafile)
testSize = len(testLabels)
testProjectedCoeff = np.zeros((testSize,k))
for i in range(testSize):
	testProjectedCoeff[i] = np.dot(test[i],principalDirections.transpose())

imgs = []
estLabels = []
for i in range(testSize) :
	trueDigit = testLabels[i]
	testImage = testProjectedCoeff[i,:]
	nnDig, nnIdx = compute_nearest_neighbors(trainProjectedCoeff,testImage)
	imgs.extend( [test[i],train[nnDig][nnIdx]] )
	estLabels.append(nnDig)

row_titles = ['Test','Nearest']
col_titles = ['%d vs. %d'%(i,j) for i,j in zip(testLabels,estLabels)]
plot_image_grid(imgs,"Image-NearestNeighbor",(28,28),testSize,2,True,row_titles=row_titles,col_titles=col_titles)

