# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:43:14 2016

@author: Ivo
"""
import numpy as np
import copy

#%% ErrorRate
def ErrorRate(Y,T):
    if Y.ndim==1 or Y.shape[0]==1:
        errors=Y!=T
        return errors.sum()/Y.size
    else: # für mehrere Ausgaben in one-hot Kodierung:
        errors=Y.argmax(0)!=T.argmax(0)
        return errors.sum()/Y.shape[1]

#%%
import nnwplot
import matplotlib.pyplot as plt

class SLN:
    def __init__(self,dIn,cOut): # Konstruktor
        self._b=np.zeros(cOut)[np.newaxis].T
        np.random.seed(42)
        self._W=np.random.randn(cOut,dIn)/np.sqrt(dIn+1)
    def neuron_mit_for(self,X):
        net=np.zeros(X.shape[1])
        for n in range(0,X.shape[1]):
            for j in range(0,X.shape[0]):
                net[n]+=self._W[j]*X[j,n]
        net+=self._b;
        return net>=0
    def neuron(self,X):
        net=self._W.dot(X)+self._b
        return net>=0
    def DeltaTrain(self, X, T, eta, maxIter, maxErrorRate):
        best = self;
        bestError = 2;
        bestIt = 0;
        N=X.shape[1]    # Anzahl Trainingsdaten
        x0 = np.ones(N)[np.newaxis]
        plt.ion() # interactive mode on
        for it in range(maxIter):
            Y = self.neuron(X)
            err = ErrorRate(Y, T)
            if (it%20) == 0:
                print('#{} {} {} {}'.format(it,self._W,self._b,err))
                nnwplot.plotTwoFeatures(X,T,self.neuron)
                #älteres python: plt.pause(0.05) # warte auf GUI event loop
            if err<bestError:
                bestError = err
                best = copy.copy(self)
                bestIt = it
            if err <= maxErrorRate:
                break
            self._W+=eta*(T-Y).dot(X.T)/N
            self._b+=eta*(T-Y).dot(x0.T)/N
        self._W=best._W
        self._b=best._b
        print('#{} {} {} {}'.format(bestIt,self._W,self._b,err))
        nnwplot.plotTwoFeatures(X,T,self.neuron)
        #älteres python: plt.pause(0.05) # warte auf GUI event loop
        return bestError, bestIt

#%% Iris-Daten Laden
iris = np.loadtxt("iris.csv",delimiter=',')
X=iris[:,0:4].T
T=iris[:,4]

#%% Test mit den Werten des vorigen Aufgabenblatts
sln=SLN(2,1)
sln._W=np.array([-1,1])
sln._b=np.array([3])

plt.figure()
nnwplot.plotTwoFeatures(X[:2,:],T,sln.neuron)

plt.figure()
nnwplot.plotTwoFeatures(X[:2,:],T,sln.neuron_mit_for)

ErrorRate(sln.neuron(X[:2,:]), T<1)
ErrorRate(sln.neuron_mit_for(X[:2,:]), T<1)

#%% UND-Daten
Xund=np.array([[0,0,1,1],[0,1,0,1]])
Tund=np.array([0,0,0,1])

#%% Training mit UND-Daten
plt.figure()
slnUND = SLN(2,1)
slnUND.DeltaTrain(Xund,Tund,0.1,100,0.01)
slnUND.neuron(Xund)

plt.figure()
nnwplot.plotTwoFeatures(Xund,Tund,slnUND.neuron)

#%% Training mit 2 Iris-Blütenarten, Merkmale 0 und 1
plt.figure()
slnIris = SLN(2,1)
slnIris.DeltaTrain(X[:2,0:100],T[0:100],0.1,200,0.0)
print(slnIris._b)
slnIris.neuron(X[:2,:])

#%% Training mit 2 Iris-Blütenarten, Merkmale 1 und 2
plt.figure()
slnIris = SLN(2,1)
slnIris.DeltaTrain(X[1:3,0:100],T[0:100],0.1,200,0.02)
slnIris.neuron(X[1:3,:])
