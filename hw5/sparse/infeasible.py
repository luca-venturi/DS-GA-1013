import numpy as np
import scipy.io

from cvxpy import *

"""
Given a possibly infeasible set of inequalities Ax <= b, returns a
vector x that tries to minimize the number of violated inequalities.
Note that finding the true minimum is a very hard problem, so try to
formulate an approximate solution that is efficiently computable.  If
the problem is feasible, your solution should return a feasible x.
Arguments:
A - 2d numpy array of shape (n,p)
b - 2d numpy array of shape (n,1)
Returns:
x - numpy array of length p that tries to make the number of
inequalities violated as small as possible
"""
def best(A,b) :
    return None
    
def run(name,A,b) :
    x = best(A,b)
    v = np.sum((np.dot(A,x)-b) > 1e-5)
    print('%s num violated = %d'%(name,int(v)))

def main() :
    data = scipy.io.loadmat('infeasible.mat')
    np.random.seed(1001)
    A1 = data['A'].astype('float64').todense()
    b1 = data['b'].astype('float64')
    run('Dataset 1',A1,b1)
    A2 = np.random.randn(1000,50)
    b2 = np.random.randn(1000,1)
    run('Dataset 2',A2,b2)
    A3 = np.random.randn(1000,50)
    b3 = np.random.randn(1000,1)**2
    run('Dataset 3',A3,b3)
    
if __name__ == "__main__" :
    main()
