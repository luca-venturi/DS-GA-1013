import numpy as np
import matplotlib.pyplot as plt

"""
Given the data points xs and ys, and x0, returns the height and slopes
of the continuous piecewise linear function that 
minimizes the square error on the data set.
Arguments:
xs: numpy array of length n containing the x coordinates of the data
ys: numpy array of length n containing the y coordinates of the data.  
Data point i is given by (xs[i],ys[i]).
x0: the x-coordinate where the piecewise linear function can change slope
Returns: the tuple (y0,alpha,beta) specifying a function f given by
f(t) = y0 + (t-x0)*alpha if t < 0 and
f(t) = y0 + (t-x0)*beta if t >= 0
"""
def fit_ls(xs,ys,x0) :
    return None #Your code here
    
def main() :
    np.random.seed(100)
    K = 20
    cxs = K*np.random.randn(5)
    cys = 10*K*np.random.randn(5)
    alps = K*np.random.randn(5)
    bets = K*np.random.randn(5)
    for i in range(5) :
        xsL = np.abs(K*np.random.randn(100))
        xsR = np.abs(K*np.random.randn(100))
        ysL = 2*K*np.random.randn(100)
        ysR = 2*K*np.random.randn(100)
        xs = np.concatenate((cxs[i]-xsL,cxs[i]+xsR))
        ys = np.concatenate((-alps[i]*xsL+ysL+cys[i],bets[i]*xsR+ysR+cys[i]))
        y0,alp,bet = fit_ls(xs,ys,cxs[i])
        print("Plot %d: alpha = %f, beta = %f, y0 = %f"%(i+1,alp,bet,y0))
        plt.scatter(xs,ys)
        plt.plot(cxs[i]-xsL,-alp*xsL+y0)
        plt.plot(xsR+cxs[i],bet*xsR+y0)
        plt.title('Data set %d'%(i+1))
        plt.savefig('Plot%d.pdf'%(i+1),bbox_inches='tight')
        plt.show()
    
if __name__ == "__main__" :
    main()
