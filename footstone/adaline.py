# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:10:34 2018

@author: Red
"""

import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    '''Min-Max normalization :(xi - min(xi))/(max(xi) - min(xi))'''
    minV = np.min(X, axis=0) * 1.0
    maxV = np.max(X, axis=0) * 1.0

    return (X * 1.0 - minV) / (maxV - minV)

def standard(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

class AdalineGD(object):
    ''' Adaline Gradient Descent Model

    Parameters
    -------------
    eta: float
        Learning rate between in (0.0 1.0)
    n_iter : int
        Passes over the training dataset. 
        Breaking dead loop if can't be seperated by a hyperplane.
    
    Attributes
    -------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    
    '''

    def __init__(self, eta=0.001, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter
        self.complex = 0 # Statistical algorithm complexity
   
    def errors(self, X, y):
        '''Statistics all errors into self.errors_'''

        predicts = self.appendedX_.dot(self.w_)
        diffs = np.where(predicts >= 0.0, 1, -1) - y
        errors = np.count_nonzero(diffs)
        self.errors_.append(errors)
        
        return errors, diffs

    def fit_adaline(self, X, y):
        ''' Fit Training data with adaptive model
        
            Parameters:
            -------------
            X: {array-like}, shape = (n_samples, n_features)
                Training vectors, where n_samples is the number of
                samples and n_features is the number of 1 sample'features.
            y: array-like, shape={n_samples}
                Target values of samples
            
            Returns:
            -------------
            self: object
        '''
        samples = X.shape[0]
        x_features = X.shape[1]
        self.w_ = 1.0 * np.zeros(1 + x_features) + 1
        #self.w_ = np.array([0, -1, 1]) * 1.0
        
        #X[:,:] = standard(X)[:,:]
        #for i in range(x_features):
        #    X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
        
        self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
        self.errors_ = []
        self.costs_ = []
        
        # record every w during whole iterations
        self.wsteps_ = []
        self.steps_ = 1  # every steps_ descent steps statistic one cose and error sample
        
        while(1):
            if (self.complex % self.steps_ == 0):
                errors, diffs = self.errors(X, y)
                self.wsteps_.append(self.w_.copy())
                cost = 1 / 2 * np.sum((y - self.appendedX_.dot(self.w_)) ** 2) 
                self.costs_.append(cost)

            self.complex += 1
            
            # minmium cost function with partial derivative wi
            deltaw = (y - self.appendedX_.dot(self.w_))
            deltaw = -self.eta * deltaw.dot(self.appendedX_)

            if np.max(np.abs(deltaw)) < 0.00001:
                print("deltaw is less than 0.00001")
                return self

            self.w_ -= deltaw
            
            if(self.complex > self.n_iter):
                print("Loops beyond n_iter %d" % self.n_iter)
                return self
        
        return self
    
    # X is a vector including features of a sample 
    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0] * 1
    
    # X is a vector including features of a sample 
    def sign(self, X):
        '''Sign function'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    # X is a vector including features of a sample 
    def predict(self, X):
        '''Return class label after unit step'''
        return self.sign(X)

    def draw_errors(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Convergence errors state")   # 收敛状态图
        plt.xlabel("Iterations")          # 迭代次数
        plt.ylabel("Miss-classifications")# 误分类错误样本数

        x = np.arange(1, 1 + len(self.errors_), 1) * self.steps_
        plt.xlim(1 * self.steps_, len(self.errors_) * self.steps_) 
        plt.ylim(0, max(self.errors_) + 1)
        
        plt.plot(x, self.errors_, c='grey')
        plt.scatter(x, self.errors_, c='black')
        plt.show()

    def draw_costs(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        if len(self.costs_) <= 1:
            print("can't plot costs for less data")
            return
        
        plt.figure()
        plt.title("Cost values J(w) state")   # 收敛状态图
        plt.xlabel("Iterations")         # 迭代次数
        plt.ylabel("Cost values J(w)")        # 代价函数值
        
        x = np.arange(1, 1 + len(self.costs_), 1) * self.steps_
        plt.xlim(1 * self.steps_, len(self.costs_) * self.steps_) 
        plt.ylim(0, max(self.costs_) + 1)
        #plt.yticks(np.linspace(0, max(self.costs_) + 1, num=10, endpoint=True))
        
        plt.plot(x, self.costs_, c='grey')
        plt.scatter(x, self.costs_, c='black')
        plt.show()
    
    def draw_vector(self, plt, V):
        ''' Draw a vector with handle plt
        
            Parameters:
            -------------
            V: {array-like}, shape = (2,) or shape = (4,)
                if shape=(2,) start point is origin (0,0)
        
        '''
        X,Y = 0,0
        
        if len(V) == 4:
            X,Y,U,V = V[0], V[1], V[2], V[3]
        else:
            U,V = V[0], V[1]

        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.004)
 
    def draw_vectors(self, figsize=(4,4)):
        if np.shape(self.w_)[0] != 3:
            print("can't draw vectors with Dimesions:%d", np.shape(self.w_)[0])
            return
        
        wvectors = np.array(self.wsteps_)
        plt.figure(figsize=figsize)
        plt.title("Adaline vectors state")  
        
        min_ = np.min(wvectors[:, 1:])
        max_ = np.max(wvectors[:, 1:])
        max_ = max(abs(min_), abs(max_)) + 0.1
        
        plt.xlim([-max_, max_])
        plt.ylim([-max_, max_])
        
        step = 1
        for i in self.wsteps_:
            self.draw_vector(plt, i[1:])
            plt.annotate('(%s)'%(step), xy=(i[1],i[2]), xytext=(0, -5),
                     textcoords = 'offset pixels', ha='left', va='top')
            step += 1

    def draw_line(self, plt, w, x1, x2, num=0, c='black'):
        # draw line: w0 + w1x1 + w2x2 = 0, x2 = -(w0 + w1x1)/w2
        if w[1] == 0 and w[2] == 0:
            print("Can't plot a line when both w1 and w2 are 0!")
            return
        elif w[2] == 0: # x1 = -w0/w1, a line vertical to x-axis
            line_x1 = [-w[0] / w[1]] * 2
            line_x2 = [-10000, np.max(x2) + 10000]
        else:
            max_ = max(np.max(np.abs(x1)),np.max(np.abs(x2))) + 1
            line_x1 = np.arange(-max_, max_ + 10, 0.5)
            line_x2 = -(line_x1 * w[1] + w[0]) / w[2]
        
        plt.plot(line_x1, line_x2, c)
        if num:
            plt.annotate('(%d)'%(num), xy=(line_x1[0],line_x2[0]), xytext=(5, -5), 
                         textcoords = 'offset pixels', ha='left', va='bottom')

    def draw_w_line(self, plt, X, step=0):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with Dimesions:%d", np.shape(self.w_)[0])
            return 
        
        steps = len(self.errors_)
        if step > steps:
            print("beyond steps number %d" % len(self.errors_))
            return
        
        x1 = X[:, 0]
        x2 = X[:, 1]
        c = str((1 - step/(steps + 1)) * 0.90)
        self.draw_line(plt, self.wsteps_[step], x1, x2, step + 1, c)

    def draw_points(self, X, labels, title='', figsize=(4,4)):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        plt.figure(figsize=figsize)
        
        plt.title(title)   
        plt.xlabel("x1")          
        plt.ylabel("x2")
        
        # x1 and x2 features
        x1 = X[:, 0]
        x2 = X[:, 1]

        max_ = abs(np.max(np.abs(X))) + 1
        
        plt.xlim(-max_, max_ + 0.5)
        plt.xticks(np.arange(-max_, max_))
        
        plt.ylim(-max_, max_)
        plt.yticks(np.arange(-max_, max_))

        plt.scatter(x1[labels == 1], x2[labels == 1], c='black', marker='o')
        plt.scatter(x1[labels == -1], x2[labels == -1], c='black', marker='s')

        for index, x, y in zip(range(len(labels)), x1, x2):
            plt.annotate('(%s,%s)'%(x,y), xy=(x,y), xytext=(-20,-20), 
                     textcoords = 'offset pixels', ha='left', va='bottom')
        
        return plt
    
    def draw_separate_line(self, X, y, title=''):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        title = "Adaline" + (' ' + title) if len(title) else ''
        plt = self.draw_points(X, y, title)
        
        # x1 and x2 features
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        self.draw_line(plt, self.w_, x1, x2, 0, c='black')

    def draw_converge_lines(self, X, y):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        plt = self.draw_points(X, y, "Perceptron convergence lines")
        
        # x1 and x2 features
        x1 = X[:, 0] 
        x2 = X[:, 1]

        self.draw_line(plt, self.w_, x1, x2, len(self.wsteps_), c='black')
        for i in range(len(self.wsteps_) - 1):
            self.draw_w_line(plt, X, i)

        plt.show()
        

class AdalineSGD(object):
    ''' Adaline Stochastic Gradient Descent Model

    Parameters
    -------------
    eta: float
        Learning rate between in (0.0 1.0)
    n_iter : int
        Passes over the training dataset. 
        Breaking dead loop if can't be seperated by a hyperplane.
    
    Attributes
    -------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    shuffle_flag : bool (default: True)
        Shuffles trainging data every epoch
        if True do prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling
        and initializing the weights.
    '''

    def __init__(self, eta=0.001, n_iter=1000,
                 shuffle=True, random_state=None):
 
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle_flag = shuffle
        if random_state:
            np.random.seed(random_state)
            
        self.complex = 0 # Statistical algorithm complexity
   
    def errors(self, X, y):
        '''Statistics all errors into self.errors_'''

        predicts = self.appendedX_.dot(self.w_)
        diffs = np.where(predicts >= 0.0, 1, -1) - y
        errors = np.count_nonzero(diffs)
        self.errors_.append(errors)
        
        self.wsteps_.append(self.w_.copy())
        cost = 1 / 2 * np.sum((y - self.appendedX_.dot(self.w_)) ** 2)
        self.costs_.append(cost)
        
        return errors, diffs
    
    def shuffle(self, X, y):
        '''Shuffle training data'''
        r = np.random.permutation(X.shape[0])
        return X[r], y[r]
    
    def update_weights(self, xi, target):
        '''Apply Adaline learning rule to update the weights'''
        deltaw = self.eta * (target - self.net_input(xi))
        self.w_[1:] += xi.dot(deltaw)
        self.w_[0] += deltaw * 1
        
        return deltaw

    def partial_fit(self, X, y):
        '''Online update w after first training'''
        if not self.w_initialized:
            self.w_ = 1.0 * np.zeros(1 + X.shape[1])
            self.w_initialized = True

        for xi, target in zip(X, y):
            self.update_weights(xi, target)

    def fit_sgd(self, X, y):
        ''' Fit Training data with adaptive model
        
            Parameters:
            -------------
            X: {array-like}, shape = (n_samples, n_features)
                Training vectors, where n_samples is the number of
                samples and n_features is the number of 1 sample'features.
            y: array-like, shape={n_samples}
                Target values of samples
            
            Returns:
            -------------
            self: object
        '''
        samples = X.shape[0]
        x_features = X.shape[1]
        self.w_ = 1.0 * np.zeros(1 + x_features) + 1
        self.w_initialized = True
        #self.w_ = np.array([0, -1, 1]) * 1.0
        
        #X[:,:] = standard(X)[:,:]
        #for i in range(x_features):
        #    X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
        
        self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
        self.errors_ = []
        self.costs_ = []
        
        # record every w during whole iterations
        self.wsteps_ = []
        self.steps_ = 1  # every steps_ descent steps statistic one cose and error sample
        
        while(1):
            self.complex += 1
            if(self.complex > self.n_iter):
                print("Loops beyond n_iter %d" % self.n_iter)
                return self

            deltaws = []
            for xi, target in zip(X, y):
                deltaw = self.update_weights(xi, target)
                deltaws.append(deltaw)
                if (self.complex % self.steps_ == 0):
                    errors, diffs = self.errors(X, y)

            if np.max(np.abs(np.array(deltaws))) < 0.0001:
                print("deltaw is less than 0.0001")
                self.wsteps_.append(self.w_.copy()) # record last w
                return self

            if self.shuffle_flag:
                X, y = self.shuffle(X, y)
                self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
            
        return self

    def fit_mbgd(self, X, y, batchn=10):
        ''' Mini Batch SGD if batchn = 1 equal fit_sgd, if batchn >= samples equal fit_adaline
        
            Parameters:
            -------------
            X: {array-like}, shape = (n_samples, n_features)
                Training vectors, where n_samples is the number of
                samples and n_features is the number of 1 sample'features.
            y: array-like, shape={n_samples}
                Target values of samples
            
            Returns:
            -------------
            self: object
        '''
        samples = X.shape[0]
        x_features = X.shape[1]
        self.w_ = 1.0 * np.zeros(1 + x_features)
        #self.w_ = np.array([0, -1, 1]) * 1.0
        
        #X[:,:] = standard(X)[:,:]
        #for i in range(x_features):
        #    X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
        
        self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
        self.errors_ = []
        self.costs_ = []
        
        # record every w during whole iterations
        self.wsteps_ = []
        self.steps_ = 1  # every steps_ descent steps statistic one cose and error sample
        
        if samples <= batchn:
            batchn = samples
        elif batchn <= 0:
            batchn = 1
        
        batch_index = 0
        batches = samples // batchn
        
        print("Fit Batch SGD with batchn %d, batches %d." % (batchn, batches))
        while(1):
            self.complex += 1
            if(self.complex > self.n_iter):
                print("Loops beyond n_iter %d" % self.n_iter)
                return self

            # minmium cost function with partial derivative wi
            for i in range(batchn):
                start_index = batch_index * i
                batchX = self.appendedX_[start_index : start_index + batchn, :]
                batchy = y[start_index : start_index + batchn]
                deltaw = -self.eta * ((batchy - batchX.dot(self.w_))).dot(batchX)

                if np.max(np.abs(deltaw)) < 0.0001:
                    print("deltaw is less than 0.0001")
                    self.wsteps_.append(self.w_.copy()) # record last w
                    return self
                
                self.w_ -= deltaw
                if (self.complex % self.steps_ == 0):
                    errors, diffs = self.errors(X, y)
                
            # must do shuffle, otherwise may lose samples at tail
            X, y = self.shuffle(X, y)
            self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
                
        return self  
    
    # X is a vector including features of a sample 
    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0] * 1
    
    # X is a vector including features of a sample 
    def sign(self, X):
        '''Sign function'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    # X is a vector including features of a sample 
    def predict(self, X):
        '''Return class label after unit step'''
        return self.sign(X)

    def draw_errors(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Convergence errors state")   # 收敛状态图
        plt.xlabel("Iterations")          # 迭代次数
        plt.ylabel("Miss-classifications")# 误分类错误样本数

        x = np.arange(1, 1 + len(self.errors_), 1) * self.steps_
        plt.xlim(1 * self.steps_, len(self.errors_) * self.steps_) 
        plt.ylim(0, max(self.errors_) + 1)
        
        plt.plot(x, self.errors_, c='grey')
        plt.scatter(x, self.errors_, c='black')
        plt.show()

    def draw_costs(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        if len(self.costs_) <= 1:
            print("can't plot costs for less data")
            return
        
        plt.figure()
        plt.title("Cost values J(w) state")   # 收敛状态图
        plt.xlabel("Iterations")         # 迭代次数
        plt.ylabel("Cost values J(w)")        # 代价函数值
        
        x = np.arange(1, 1 + len(self.costs_), 1) * self.steps_
        plt.xlim(1 * self.steps_, len(self.costs_) * self.steps_) 
        plt.ylim(0, max(self.costs_) + 1)
        #plt.yticks(np.linspace(0, max(self.costs_) + 1, num=10, endpoint=True))
        
        plt.plot(x, self.costs_, c='grey')
        plt.scatter(x, self.costs_, c='black')
        plt.show()
    
    def draw_vector(self, plt, V):
        ''' Draw a vector with handle plt
        
            Parameters:
            -------------
            V: {array-like}, shape = (2,) or shape = (4,)
                if shape=(2,) start point is origin (0,0)
        
        '''
        X,Y = 0,0
        
        if len(V) == 4:
            X,Y,U,V = V[0], V[1], V[2], V[3]
        else:
            U,V = V[0], V[1]

        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.004)
 
    def draw_vectors(self, figsize=(4,4)):
        if np.shape(self.w_)[0] != 3:
            print("can't draw vectors with Dimesions:%d", np.shape(self.w_)[0])
            return
        
        wvectors = np.array(self.wsteps_)
        plt.figure(figsize=figsize)
        plt.title("Adaline vectors state")  
        
        min_ = np.min(wvectors[:, 1:])
        max_ = np.max(wvectors[:, 1:])
        max_ = max(abs(min_), abs(max_)) + 0.1
        
        plt.xlim([-max_, max_])
        plt.ylim([-max_, max_])
        
        step = 1
        for i in self.wsteps_:
            self.draw_vector(plt, i[1:])
            plt.annotate('(%s)'%(step), xy=(i[1],i[2]), xytext=(0, -5),
                     textcoords = 'offset pixels', ha='left', va='top')
            step += 1

    def draw_line(self, plt, w, x1, x2, num=0, c='black'):
        # draw line: w0 + w1x1 + w2x2 = 0, x2 = -(w0 + w1x1)/w2
        if w[1] == 0 and w[2] == 0:
            print("Can't plot a line when both w1 and w2 are 0!")
            return
        elif w[2] == 0: # x1 = -w0/w1, a line vertical to x-axis
            line_x1 = [-w[0] / w[1]] * 2
            line_x2 = [-10000, np.max(x2) + 10000]
        else:
            max_ = max(np.max(np.abs(x1)),np.max(np.abs(x2))) + 1
            line_x1 = np.arange(-max_, max_ + 10, 0.5)
            line_x2 = -(line_x1 * w[1] + w[0]) / w[2]
        
        plt.plot(line_x1, line_x2, c)
        if num:
            plt.annotate('(%d)'%(num), xy=(line_x1[0],line_x2[0]), xytext=(5, -5), 
                         textcoords = 'offset pixels', ha='left', va='bottom')

    def draw_w_line(self, plt, X, step=0):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with Dimesions:%d", np.shape(self.w_)[0])
            return 
        
        steps = len(self.errors_)
        if step > steps:
            print("beyond steps number %d" % len(self.errors_))
            return
        
        x1 = X[:, 0]
        x2 = X[:, 1]
        c = str((1 - step/(steps + 1)) * 0.90)
        self.draw_line(plt, self.wsteps_[step], x1, x2, step + 1, c)

    def draw_points(self, X, labels, title='', figsize=(4,4)):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        plt.figure(figsize=figsize)
        
        plt.title(title)   
        plt.xlabel("x1")          
        plt.ylabel("x2")
        
        # x1 and x2 features
        x1 = X[:, 0]
        x2 = X[:, 1]

        max_ = abs(np.max(np.abs(X))) + 1
        
        plt.xlim(-max_, max_ + 0.5)
        plt.xticks(np.arange(-max_, max_))
        
        plt.ylim(-max_, max_)
        plt.yticks(np.arange(-max_, max_))

        plt.scatter(x1[labels == 1], x2[labels == 1], c='black', marker='o')
        plt.scatter(x1[labels == -1], x2[labels == -1], c='black', marker='s')
        
        if len(X) < 100:
            for index, x, y in zip(range(len(labels)), x1, x2):
                plt.annotate('(%s,%s)'%(x,y), xy=(x,y), xytext=(-20,-20), 
                         textcoords = 'offset pixels', ha='left', va='bottom')
        
        return plt
    
    def draw_separate_line(self, X, y, title=''):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        title = "Adaline" + (' ' + title) if len(title) else ''
        plt = self.draw_points(X, y, title)
        
        # x1 and x2 features
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        self.draw_line(plt, self.w_, x1, x2, 0, c='black')

    def draw_converge_lines(self, X, y):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        plt = self.draw_points(X, y, "Perceptron convergence lines")
        
        # x1 and x2 features
        x1 = X[:, 0] 
        x2 = X[:, 1]

        self.draw_line(plt, self.w_, x1, x2, len(self.wsteps_), c='black')
        for i in range(len(self.wsteps_) - 1):
            self.draw_w_line(plt, X, i)

        plt.show()

def irisTrainSGD(type=0):
    import pandas as pd
    '''
    Columns Information:
       1. sepal length in cm
       2. sepal width in cm
       3. petal length in cm
       4. petal width in cm
       5. class: 
          -- Iris Setosa
          -- Iris Versicolour
          -- Iris Virginica

    '''
    df = pd.read_csv('db/iris/iris.data', header=0)
    
    # get the classifications
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    
    # get samples' features 2(sepal width) and 4(petal width)
    X = df.iloc[0:100, [1,3]].values
    '''
    # get samples' features 2(sepal length) and 4(petal length)
    X = df.iloc[0:100, [0,2]].values
    '''

    if type == 1:
        irisPerceptron = AdalineSGD(0.001, 40, 1)
        irisPerceptron.fit_sgd(X, y)
    elif type == 0:
        irisPerceptron = AdalineGD(0.001, 40)
        irisPerceptron.fit_adaline(X, y)
    elif type == 2:
        irisPerceptron = AdalineSGD(0.001, 40)
        irisPerceptron.fit_mbgd(X, y)

    #irisPerceptron.draw_errors()
    #irisPerceptron.draw_separate_line(X, y, 'iris')
    #irisPerceptron.draw_converge_lines(X, y)
    #irisPerceptron.draw_vectors()
    irisPerceptron.draw_costs()
    print("LastCost: %f" % irisPerceptron.costs_[-1])
    print('Weights: %s' % irisPerceptron.w_)
    #print('Steps: %d' % len(irisPerceptron.errors_))
    #print('Complex: %d' % irisPerceptron.complex)

def boolOrTrain():
    # Bool or Train         x11 x12 y1
    BoolOrTrain = np.array([[0, 0, -1],
                            [0, 1,  1],
                            [1, 0,  1],
                            [1, 1,  1]])
    X = BoolOrTrain[:, 0:-1]
    y = BoolOrTrain[:, -1]
    
    BoolOr = AdalineSGD(0.01, 10000)
    BoolOr.fit_sgd(X, y)

    print('Weights: %s' % BoolOr.w_)
    print('Errors: %s' % BoolOr.errors_)
    #BoolOr.draw_separate_line(X, y,'Bool-or')
    #BoolOr.draw_vectors()
    BoolOr.draw_costs()
    BoolOr.draw_converge_lines(X,y)

#irisTrainSGD(1)
irisTrainSGD(2)
