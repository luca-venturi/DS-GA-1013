"""
Tools for loading the MNIST Data.

@author: Brett
"""

import numpy as np
import scipy as sp
from mnist_tools import *
from plot_tools import *

"""
Given a 2-dimensional numpy array trainImages of 
training images with shape (n,d)
(each row is an image), denoise the given noisyImage by orthogonally 
projecting it onto the row space of trainImages.
"""
def denoise(trainImages, noisyImage) :
    #Your code here
    return None

"""
Assumes the data file is in 'mnist_all.mat'.
"""
def main() :
    datafile = "mnist_all.mat" #Change if you put the file in a different path
    train = load_train_data(datafile)
    test,noisyTest,testLabels = load_noisy_test_data(datafile)
    imgs = []
    for i in range(len(testLabels)) :
        trueDigit = testLabels[i]
        testImage = test[i,:]
        noisyImage = noisyTest[i,:]
        denoisedImage = denoise(train[trueDigit],noisyImage)
        imgs.extend([testImage,noisyImage,denoisedImage])
    plot_image_grid(imgs,
                    "Image-Noisy-Denoised",
                    (28,28),len(testLabels),3,True,row_titles=['True','Noisy','Denoised'])

if __name__ == "__main__" :
    main()
