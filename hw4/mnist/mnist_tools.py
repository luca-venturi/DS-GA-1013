"""
Tools for loading the MNIST Data.

@author: Brett
"""

import scipy.io
import numpy as np

_mfile = None
train = []
test = []

def _load_data(filename) :
    global _mfile
    if _mfile is None :
        _mfile = scipy.io.loadmat(filename)

def _load_digit_data(digit,typename) :
    global _mfile
    data = _mfile[typename+str(digit)].astype('float64')
    return data

"""
(Internal function not used by students.)
Returns a tuple (train,test).  Each element is a 2d numpy array where each
row has length 785.  The first 784 values are the pixels of a 28x28 image
and the last digit is what digit is represented.
filename contains 
the name of the file containing the MNIST data (as a Matlab .mat file).
"""
def _load_mnist_data(filename) :
    global train, test
    _load_data(filename)
    np.random.seed(777)
    if len(train) == 0 :
        trlist = []
        telist = []
        for i in range(10) :
            if i == 5 :
                tmp = _load_digit_data(i,"train")[:4500,:]
            else :
                tmp = _load_digit_data(i,"train")[:500,:]
            trlist.append(np.concatenate((tmp,i*np.ones((tmp.shape[0],1))),axis=1))
            if i == 5 :
                tmp = _load_digit_data(i,"test")[:900,:]
            else :
                tmp = _load_digit_data(i,"test")[:100,:]
            telist.append(np.concatenate((tmp,i*np.ones((tmp.shape[0],1))),axis=1))
        train = np.concatenate(trlist)
        test = np.concatenate(telist)
        np.random.shuffle(train)
        np.random.shuffle(test)
    return (train,test)

"""
Returns a 2d numpy array of training data formatted as described
in the comments for _load_mnist_data.
filename contains the name of the file containing the 
MNIST data (as a Matlab .mat file).
"""
def load_train_data(filename) :
    (train,test) = _load_mnist_data(filename)
    return train

"""
Returns a 2d numpy array of testing data formatted as described
in the comments for _load_mnist_data.
filename contains the name of the file containing the 
MNIST data (as a Matlab .mat file).
"""
def load_test_data(filename) :
    (train,test) = _load_mnist_data(filename)
    return test
