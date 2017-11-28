import csv
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

"""
Finds the non-negative least squares solution b for given data X,y.
More precisely, it finds the vector b with non-negative entries that
minimizes the 2-norm ||Xb-y||.
Arguments:
X - 2d numpy array of shape (n,p)
y - 1d numpy array of length n
Returns:
A 1d numpy array b of length p with non-negative entries that
minimizes ||Xb-y||
"""
def nnls(X,y) :
    return None #Your code here

def main() :
    with open('Prostate.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = []
        for row in reader:
            rows.append(row[1:])
            names = rows[0]
            arr = np.array(rows[1:],dtype='float64')
    
    print('Coeff Names',names)

    np.random.seed(1000)    
    np.random.shuffle(arr)

    X,y = arr[:,:-1],arr[:,-1]
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    mean,std = np.mean(X,axis=0),np.std(X,axis=0)
    std[std==0] = 1
    mean[0] = 0
    X = (X-mean)/std
    coeffs = nnls(X,y)
    print('Coeffs are ')
    for i in range(len(coeffs)) :
        print(coeffs[i,0])

if __name__ == "__main__" :
    main()
