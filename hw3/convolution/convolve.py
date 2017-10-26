from mnist_tools import *
from plot_tools import *
import numpy as np
from scipy.ndimage.filters import convolve
"""
Given an image img perform the following steps:
1) Convolve with the edge detection kernel
2) Let M denote the maximum value over all
values in the resulting convolved image.
3) Threshold the resulting convolution so that all values in the
image smaller than .25*M are set to 0, and all values larger than 
.25*M are set to 1.
4) Return the resulting thresholded convolved image.
"""
def treshold(mat,eps=0.25) :
	treshold = eps * np.amax(mat)	
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			mat[i,j] = np.float(mat[i,j] > treshold)
	return mat

def edge_detect(img) :
	eps = 0.125
	x = np.asarray([[-eps,-eps,-eps],[-eps,1.,-eps],[-eps,-eps,-eps]])
	return treshold(convolve(img,x,mode='constant',cval=0.,origin=0))
    
def main() :
    test,testLabels = load_test_data("mnist_all.mat")
    imgs = []
    for im in test :
        img = im.reshape((28,28))
        imgs.extend([img,edge_detect(img)])
    plot_image_grid(imgs,"MNistConvolve",bycol=True,
                    row_titles=['True','Convolved'])

if __name__ == "__main__" :
    main()
