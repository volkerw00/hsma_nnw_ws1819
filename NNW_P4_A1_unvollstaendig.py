# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:43:14 2016

@author: Ivo
"""
import numpy as np
import nnwplot
import matplotlib.pyplot as plt

#%% ErrorRate
def ErrorRate(Y,T):
    if Y.ndim==1 or Y.shape[0]==1:
        errors=Y!=T
        return errors.sum()/Y.size
    else: # für mehrere Ausgaben in one-hot Kodierung:
        errors=Y.argmax(0)!=T.argmax(0)
        return errors.sum()/Y.shape[1]
        
#%%
class MLN:
    def __init__(self,dIn,hidden,cOut): # Konstruktor
        np.random.seed(42)
        self._b1=np.zeros((hidden,1))
        self._W1=np.random.randn(hidden,dIn)/np.sqrt(dIn)
        self._b2=np.zeros((cOut,1))
        self._W2=np.random.randn(cOut,hidden)/np.sqrt(dIn)
        if cOut==1:
            self.neuron=self.ythreshold
        else:
            self.neuron=self.ythresholdMult
    def net1(self,X):
         return self._W1.dot(X)+self._b1
    def z(self,X):
         return np.tanh(self.net1(X))
    def net2(self,Z):
         return self._W2.dot(Z)+self._b2
    def y(self,X):
        return self.net2(self.z(X))

    def ythreshold(self,X):
        return self.y(X)>=0
    
    def onehot(self,T):
        e=np.identity(self._W2.shape[0])
        return e[:,T.astype(int)]
    def ythresholdMult(self,X):
        return self.onehot(self.y(X).argmax(0))

    def backprop(self, X, T, Y, Z):
        N = X.shape[1] 
        deltak = Y-T
        deltaW2 = deltak.dot(Z.T)
        deltaB2 = deltak.dot(np.ones([1,N]).T); # entspricht np.sum(deltak,1)
  
        deltaj = (1-Z**2)*(self._W2.T.dot(deltak));
        # z: m*N, W2: c*m, deltak:c*N; das deltaj für jedes n soll sein 
        # zugehöriges (1-z_j^2) abbekommen
        deltaW1 = deltaj.dot(X.T);
        deltaB1 = deltaj.dot(np.ones([1,N]).T); # entspricht np.sum(deltaj,1)
        return deltaW1, deltaB1, deltaW2, deltaB2
    
    def DeltaTrain(self, X, T, eta, maxIter, maxErrorRate):
        best = self;
        bestError = 2;
        bestIt = 0;
        N=X.shape[1]    # Anzahl Trainingsdaten
        x0 = np.ones(N)[np.newaxis]
        plt.ion() # interactive mode on
        for it in range(maxIter):
            Z = self.z(X)
            Y = self.neuron(X)
            err = ErrorRate(Y, T)
            if (it%20) == 0:
                print('#{} err:{}\n{}\n{}\n{}\n{}'.format(it,err,self._W1,self._b1,self._W2,self._b2))
                nnwplot.plotTwoFeatures(X,T,self.neuron)
                #älteres python: plt.pause(0.05) # warte auf GUI event loop
            if err<bestError:
                bestError = err
                best = copy.copy(self)
                bestIt = it
            if err <= maxErrorRate:
                break
            deltaW1, deltaB1, deltaW2, deltaB2 = self.backprop(X,T,Y,Z)
            self._W1-=eta*deltaW1/N
            self._b1-=eta*deltaB1/N
            self._W2-=eta*deltaW2/N
            self._b2-=eta*deltaB2/N
        ergaenzen Sie hier den Update der Gewichte
        print('#{} err:{}\n{}\n{}\n{}\n{}'.format(it,err,self._W1,self._b1,self._W2,self._b2))
        nnwplot.plotTwoFeatures(X,T,self.neuron)
        #älteres python: plt.pause(0.05) # warte auf GUI event loop
        return bestError, bestIt

#%% Iris-Daten Laden
iris = np.loadtxt("iris.csv",delimiter=',')
X=iris[:,0:4].T
T=iris[:,4]

#%% Training mit 2 Iris-Blütenarten, Merkmale 0 und 1
plt.figure()
slnIris = MLN(2,1,1) # HU: 1 und 5 gut, 2,3,8 schlecht
slnIris.DeltaTrain(X[:2,0:100],T[0:100],0.01,2000,0.00)
slnIris.neuron(X[:2,:])
#%% Training mit 3 Iris-Blütenarten, Merkmale 2 und 3
plt.figure()
slnIris = MLN(2,5,3)
slnIris.DeltaTrain(X[2:4,:],slnIris.onehot(T),0.01,2000,0.04)
slnIris.neuron(X[2:4,:]).argmax(0)

