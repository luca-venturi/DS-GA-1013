import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from plot_tools import *
from cvxpy import *

"""
Returns a denoised image x obtained by finding the closest
image to the given noisy image while penalizing differences of each
pixel with its neighbors.
Arguments:
img - 2d numpy array of shape (64,64) containing the noisy image
lam - the parameter governing how much to penalize neighbor
differences
Returns:
The denoised image 
"""
def denoise(img,lam=1) :
    return None #Your code here
    
def main() :
    np.random.seed(1000)
    data = scipy.io.loadmat('olivettifaces.mat')['faces'].T.astype('float64')
    imgs = []
    for i in range(6,10) :        
        image = data[10*i+6,:].astype('float64')
        image = image.reshape(64,64).T
        s = 10
        noise = s*np.random.randn(*image.shape)
        noisyimage = image+noise
        denoised = denoise(noisyimage,lam=0.1)
        denoised1 = denoise(noisyimage,lam=1)
        denoised10 = denoise(noisyimage,lam=10)
        imgs.extend([image,noisyimage,denoised,denoised1,denoised10])
    plot_image_grid(imgs,'denoise',n_col=5,n_row=4,
                    col_titles=['True','Noisy','$\lambda$=0.1'
                    ,'$\lambda$=1','$\lambda$=10'])

if __name__ == "__main__" :
    main()
