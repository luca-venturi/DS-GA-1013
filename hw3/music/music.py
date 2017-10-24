import numpy as np
import matplotlib.pyplot as plt

"""
Given samples f(0),...,f(n-1) in a numpy array v, plots the periodogram.
You are also given the coefficients c and the frequencies t so you can superimpose
a stem plot of the points (t_i,|c_i|^2) on top of the periodogram.
Arguments:
v: numpy array containing the samples of f
c: true coefficients
t: true (fractional) frequencies in [-1/2,1/2)
Returns: Nothing, but plots the periodogram and the points (t_i,|c_i|^2)
"""
def periodogram(v,c,t) :
    return None
    
"""
Given samples f(0),...,f(n-1) in a numpy array v, plots the pseudospectrum.
You are also given the number of frequencies s,
the coefficients c and the frequencies t so you can superimpose
a stem plot of the points (t_i,|c_i|) on top of the pseudospectrum.
Arguments:
v: numpy array containing the samples of f
s: the number of frequencies
c: true coefficients
t: true (fractional) frequencies in [-1/2,1/2)
Returns: Nothing, but plots the pseudospectrum and the points (t_i,|c_i|)
"""
def music(v,s,c,t) :
    return None
    
def main() :
    np.random.seed(5)
    s = 5                     
    cs = (np.random.rand(s)-.5+1.0j*(np.random.rand(s)-.5))
    ts = [-.25,-.21,-.075,.4,.49]
    N = 100
    v = np.zeros(N,dtype='complex128')
    for i in range(N) :
        for j in range(len(ts)) :
            v[i] += cs[j]*np.exp(2.0j*np.pi*i*ts[j])

    periodogram(v,cs,ts)
    music(v,s,cs,ts)
    
if __name__ == "__main__" :
    main()
