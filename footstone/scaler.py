# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:10:34 2018

@author: Red
"""

import numpy as np
import matplotlib.pyplot as plt

def draw_points(X, labels, title='', figsize=(4,4), coordinate=False):
    plt.figure(figsize=figsize)
    
    plt.title(title)   
    plt.xlabel("x1")          
    plt.ylabel("x2")
    
    # x1 and x2 features
    x1 = X[:, 0]
    x2 = X[:, 1]

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    max,min = np.max(labels), np.min(labels)
    plt.scatter(x1[labels == max], x2[labels == max], c='black', marker='o')
    plt.scatter(x1[labels == min], x2[labels == min], c='black', marker='o')

    circle = plt.Circle((0, 0), radius=1.1, fill=False, color='red')
    plt.gca().add_patch(circle)
    
    if coordinate:
        for index, x, y in zip(range(len(labels)), x1, x2):
            plt.annotate('(%.2f,%.2f)'%(x,y), xy=(x,y), xytext=(-20,-20), 
                     textcoords = 'offset pixels', ha='left', va='bottom')
    
    return plt

# generate noraml distribution train set
def normal_dis_trainset(positive=100, negtive=100, type='normal'):
    np.random.seed(0)
    
    if type == 'normal':
        numA = np.random.normal(3, 2, (2, positive))
        numB = np.random.normal(-6, 2, (2, negtive))
    elif type == 'ones':
        numA = np.ones((2, positive)) - 3
        numB = np.ones((2, negtive)) + 5
    else:
        numA = np.zeros((2, positive)) - 3
        numB = np.zeros((2, negtive)) + 5

    Ax, Ay = numA[0] * 0.5, numA[1]
    Bx, By = numB[0], numB[1]
    
    labels = np.zeros((negtive + positive, 1))
    trainset = np.zeros((negtive + positive, 2))
    trainset[0:positive,0] = Ax[:]
    trainset[0:positive,1] = Ay[:]
    labels[0:positive] = 1
    
    trainset[positive:,0] = Bx[:]
    trainset[positive:,1] = By[:]
    labels[positive:] = -1

    return trainset, labels.reshape(positive + negtive,)

# normalization into scope [0-1]
def normalize(X):
    '''Min-Max normalization :(xi - min(xi))/(max(xi) - min(xi))'''
    minV = np.min(X, axis=0) * 1.0
    maxV = np.max(X, axis=0) * 1.0

    return (X * 1.0 - minV) / (maxV - minV)

# Mean-subtraction, move data around origin (0,0...)
def zero_centered(X):
    return X - np.mean(X, axis=0)

# Mean is 0, σ is 1
def standard(X):
    std = np.std(X, axis=0)
    #std[std == 0] = 1e-3
    return zero_centered(X) / std

def shuffle(X, y, seed=None):
    idx = np.arange(X.shape[0])
    
    np.random.seed(seed)
    np.random.shuffle(idx)
    
    return X[idx], y[idx]

class DataScaler():
    def __init__(self, X_train):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        
        self.mms = MinMaxScaler()
        self.mms.fit(X_train)
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

    def sklearn_normalize(self, X):
        return self.mms.transform(X)

    def sklearn_standard(self, X):
        return self.scaler.transform(X)

if __name__ == "__main__":
    X,y = normal_dis_trainset(10, 10, 'normal')
    '''
    X[:, 1] *= 1.1
    print(y)
    draw_points(X,y)
    X1 = standard(X)
    draw_points(X1,y)
    plt.show()
    '''
    X,y = shuffle(X, y)
    print(y)
