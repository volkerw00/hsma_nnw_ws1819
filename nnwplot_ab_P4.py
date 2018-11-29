# based on: https://github.com/eakbas/tf-svm/blob/master/plot_boundary_on_data.py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def plotTwoFeatures(X,T,pred_func):
    if X.ndim!=2:
        raise ValueError('X be a matrix (2 dimensional array).')
    tp=False
    if X.shape[0]!=2 and X.shape[1]==2: 
        X=X.T
        tp=True
    if X.shape[0]!=2:
        raise ValueError('X must contain exactly 2 features.')
        
    # determine canvas borders
    mins = np.amin(X,1); 
    mins = mins - 0.1*np.abs(mins);
    maxs = np.amax(X,1); 
    maxs = maxs + 0.1*maxs;

    ## generate dense grid
    xs,ys = np.meshgrid(np.linspace(mins[0],maxs[0],300), 
            np.linspace(mins[1], maxs[1], 300));


    # evaluate model on the dense grid
    try:
        Z = pred_func(np.c_[xs.flatten(), ys.flatten()].T);
    except:
        Z = pred_func(np.c_[xs.flatten(), ys.flatten()]);

    if tp:
        Z=Z.T
    
    if Z.ndim>1 and Z.shape[0]>1: # onehot? -> convert
        Z=Z.argmax(0)
    Z = Z.reshape(xs.shape)
    
    if T.ndim>1 and T.shape[0]>1: # onehot? -> convert
        T=T.argmax(0)
        
    # Plot the contour and training examples
    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0,:], X[1,:], c=T, s=50,
            cmap=colors.ListedColormap(['orange', 'blue', 'green']))
    plt.show()
