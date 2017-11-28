import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lin
from cvxpy import *

"""
Given vectors vs[i,:] and scalars bs[i], returns the center and radius
of the largest disc (circle including interior) contained in the
region {x : dot(vs[i,:],x) >= bs[i]}.  The inputs will guarantee such
a disc will exist.
Arguments:
vs - 2d numpy array of shape (n,2)
bs - 2d numpy array of shape (n,1)
Returns:
(x,r) where x is the center and r is the radius of the largest disc
satisfying the constraints
x - numpy vector of length 2
r - float
"""
def disc(vs,bs) :
    return None #Your code here

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])*1.0/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])*1.0/(p2[0]-p1[0])*(xmin-p1[0])

    l = lin.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def plot(name,vs,bs,box) :
    plt.xlim([box[0],box[1]])
    plt.ylim([box[2],box[3]])
    ax = plt.gca()
    for i in range(vs.shape[0]) :
        v,b = vs[i,:],bs[i]
        p0 = [b/v[0],0] if v[0] != 0 else [0,b/v[1]]
        p1 = [p0[0]+v[1],p0[1]-v[0]]
        newline(p0,p1)
    xc,r = disc(vs,bs)
    ax.add_artist(plt.Circle(xc,r,color='r'))
    plt.title('%s: r = %0.3f'%(name,r))
    plt.savefig(name+'.pdf',bbox_inches='tight')
    plt.close()

def main() :
    box1 = [-1,2,-1,2]
    vs1 = np.array([[0,1],[1,0],[-1,0],[0,-1]])
    bs1 = np.array([0,0,-1,-1]).reshape((4,1))
    plot('fig1',vs1,bs1,box1)
    box2 = [-1,6,-6,7]
    vs2 = np.array([[1,0],[1,-1],[1,1],[-1,0]])
    bs2 = np.array([0,-1,0,-5]).reshape((4,1))
    plot('fig2',vs2,bs2,box2)
    box3 = [-1,6,-6,7]
    vs3 = np.array([[1,0],[1,-1],[1,1],[-1,-1],[-1,5]])
    bs3 = np.array([0,-1,0,-4,-7]).reshape((5,1))
    plot('fig3',vs3,bs3,box3)
    np.random.seed(1000)
    box4 = [-10,10,-10,10]
    vs4 = -(np.random.rand(100,2)*20-10)
    bs4 = (-np.linalg.norm(vs4,axis=1)**2).reshape(100,1)
    plot('fig4',vs4,bs4,box4)
    
if __name__ == "__main__" :
    main()
