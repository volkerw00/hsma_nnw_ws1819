# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:43:14 2016

@author: Ivo
"""
import numpy as np

#%% IRIS numpy
iris = np.loadtxt("iris.csv",delimiter=',')
X=iris[:,0:4]
T=iris[:,4]
X
T
#%% X transponieren, damit Daten in Zeilen
X=X.T
#%% matplotlib
import matplotlib.pyplot as plt

plt.figure(2) # Öffne/aktiviere Plot-Fenster mit der Nummer 2
plt.clf()     # lösche aktuelles Plot-Fenster
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Plot the training points
# plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.prism)
# Alternative:
from matplotlib import colors
plt.scatter(X[0,:], X[1,:], c=T, cmap=colors.ListedColormap(['red', 'green', 'blue']))

#%%
def neuron(X):
    net=np.zeros(X.shape[1])
    W=[-1,1]
    for n in range(0,X.shape[1]):
        for j in range(0,X.shape[0]):
            net[n]+=W[j]*X[j,n]
    return net>=-3
#%%
import nnwplot
nnwplot.plotTwoFeatures(X[:2,:],T,neuron)

#%% Neues Plot-Fenster Für die letzten beiden Merkmale (als Beispiel)
plt.figure()
nnwplot.plotTwoFeatures(X[-2:,:],T,neuron)

#%% Variante mit closure
def neuronWTH(W,TH):
    def neuron(X):
        N=X.shape[1] # Anzahl Daten/Merkmalsvektoren
        net=np.zeros(N)
        for n in range(N):
            for j in range(0,X.shape[0]):
                net[n]+=W[j]*X[j,n]
        return net>=TH  
    return neuron
#%%
def plotneuronWTH(W,TH):
    plt.figure(); nnwplot.plotTwoFeatures(X[:2,:],T,neuronWTH(W,TH)); plt.title('W: {} TH: {}'.format(W, TH));
    
#%%
plotneuronWTH([-0.3,1],1)
plotneuronWTH([-0.3,1],2)
plotneuronWTH([-0.3,1],3)

plotneuronWTH([-0.2,1],2)
plotneuronWTH([-0.1,1],2)
plotneuronWTH([-0,1],2)

plotneuronWTH([-0.3,1],1)
plotneuronWTH([-0.3,1],2)
plotneuronWTH([-0.3,1],3)

plotneuronWTH([-0.2,1],2)
plotneuronWTH([-0.1,1],2)
plotneuronWTH([-0,1],2)

#%% Weitere Versuche
plotneuronWTH([-0.3,1],1)
plotneuronWTH([-0.5,1],1)
plotneuronWTH([-0.7,1],1)
plotneuronWTH([-1,1],0)
plotneuronWTH([-1,1],-1)
plotneuronWTH([-1,1],-2)
plotneuronWTH([-1,1],-3)
plotneuronWTH([-1.2,1],-3)
plotneuronWTH([-1.4,1],-3)

#%% 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0,:], X[1,:], X[2,:], c=T, cmap=colors.ListedColormap(['red', 'green', 'blue']))
