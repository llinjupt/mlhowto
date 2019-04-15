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

def data_split(X, y, ratio=0.3, random_state=0):
    from sklearn.model_selection import train_test_split
    
    # 'X_train, y_test, x_labels, y_labels = '
    return train_test_split(X, y, test_size=ratio, random_state=random_state)

if __name__ == "__main__":
    X,y = normal_dis_trainset(3, 3)
    
    X_train, X_test, y_train, y_test = data_split(X, y)
    print(y_train)
    print(y_test)
    