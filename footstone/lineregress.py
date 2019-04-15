# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:58:42 2018

@author: Red
"""

import numpy as np
import dbload
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class LRegress():
    def __init__(self, b=None, w1=None, eta=0.001, tol=0.001):
        self.eta = eta
        self.tol = tol
        
        np.random.seed(None)
        self.b = b
        if self.b is None:
            self.b = np.random.randn(1)[0]
        
        self.w1 = w1
        if self.b is None:
            self.w1 = np.random.randn(1)[0]
    
    # both w and b is verctor, and X is 2D array 
    def hypothesis(self, X):
        return self.b + self.w1 * X[:,0]

    def predict(self, X):
        try:
            X = (X - self.mean) / self.std
        finally:
            print(X)
            return self.hypothesis(X)

    # MSE/LSE Least square method
    def cost(self, X, y):
        return np.sum((self.hypothesis(X) - y)**2) / X.shape[0] / 2

    def delta_b(self, X, y):
        return np.sum(self.b + self.w1*X[:,0] - y) / X.shape[0]

    def delta_w(self, X, y):
        derective = (self.b + self.w1*X[:,0] - y) * X[:,0]
        return np.sum(derective) / X.shape[0]
    
    def standard(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        assert(np.std(X, axis=0).any())
        return (X - self.mean) / self.std

    def bgd(self, X, y, max_iter=1000, standard=True):
        # for drawing Gradient Decent Path
        self.costs_ = [] 
        self.bs_ = []
        self.w1s_ = []
        
        self.steps_ = 1
        self.complex = 0
        
        if standard: X = self.standard(X)
        for loop in range(max_iter):
            cost = self.cost(X, y)
            if(cost < self.tol):
                print("cost reduce very tiny less than tol, just quit!")
                return X
            
            delta_b  = self.eta * self.delta_b(X, y)
            delta_w1 = self.eta * self.delta_w(X, y)
            
            # update weights and b together
            if self.complex % self.steps_ == 0:
                self.bs_.append(self.b)
                self.w1s_.append(self.w1)
                cost = self.cost(X,y)
                self.costs_.append(cost)

            self.b -= delta_b
            self.w1 -= delta_w1
            self.complex += 1
        
        # return standard X
        return X
    
    def sgd(self, X, y, max_iter=1000, standard=True):
        # for drawing Gradient Decent Path
        self.costs_ = [] 
        self.bs_ = []
        self.w1s_ = []
        
        self.steps_ = 1
        self.complex = 0
        
        STDX = self.standard(X) if standard else X
        import scaler
        X,y = scaler.shuffle(STDX, y)        
        for loop in range(max_iter):
            for Xi, yi in zip(X, y):
                Xi = Xi.reshape(Xi.size, 1)
                cost = self.cost(Xi, yi)
                if(cost < self.tol):
                    print("cost reduce very tiny less than tol, just quit!")
                    return STDX

                delta_b  = self.eta * self.delta_b(Xi, yi)
                delta_w1 = self.eta * self.delta_w(Xi, yi)
                
                self.b -= delta_b
                self.w1 -= delta_w1
    
                # update weights and b together
                if self.complex % self.steps_ == 0:
                    self.bs_.append(self.b)
                    self.w1s_.append(self.w1)
                    cost = self.cost(X,y)
                    self.costs_.append(cost)
                self.complex += 1
                
        return STDX
    
    def draw_costs(self):
        '''Draw errors info with matplotlib'''
        if len(self.costs_) <= 1:
            print("can't plot costs for less data")
            return

        plt.figure()
        plt.title("Cost values J(w) state")
        plt.xlabel("Iterations")             
        plt.ylabel("Cost values J(w)")     
        
        x = np.arange(1, 1 + len(self.costs_), 1) * self.steps_
        plt.xlim(1 * self.steps_, len(self.costs_) * self.steps_) 
        plt.ylim(0, max(self.costs_) + 1)

        plt.plot(x, self.costs_, c='grey')
        plt.scatter(x, self.costs_, c='black')
       
    def draw_points(self, X, y, title='', coordinate=False):
        plt.figure()
        
        plt.title(title)   
        plt.xlabel("BMI") #'x1'
        plt.ylabel("Fat%")  #'y'
        
        # x1 and x2 features
        x1 = X[:,0]
        x2 = y

        plt.scatter(x1, x2, c='black', marker='o')
        if coordinate:
            for index, x, y in zip(range(X.shape[0]), x1, x2):
                plt.annotate('(%.2f,%.2f)'%(x,y), xy=(x,y), xytext=(-20,-20), 
                         textcoords = 'offset pixels', ha='left', va='bottom')
        return plt

    def draw_line(self, plt, x1, y, c='black'):
        # draw line: h(x1) = b + w1x1
        if self.b == 0 and self.w1 == 0:
            print("Can't plot a line when both w1 and w2 are 0!")
            return
        elif self.w1 == 0: # x1 = -b/w1, a line vertical to x-axis
            line_x1 = [-self.b / self.w1] * 2
            line_x2 = [-100, 100]
        else:
            max_ = np.max(x1) 
            min_ = np.min(x1) 
            
            line_x1 = np.arange(min_ - 1, max_ + 1, 0.5)
            line_x2 = line_x1 * self.w1 + self.b

        plt.plot(line_x1, line_x2, c)
        
    def draw_vertical_line(self, plt, X, y):
        for i in range(5):
            idx = (np.random.randint(X.shape[0], size=1))
            x = X[idx][0]
            plt.plot([x, x], [y[idx], self.hypothesis(X[idx])], c='blue')

    def draw_separate_line(self, X, y, title='', vertical=False):
        title = "Linear Regression" + (' ' + title) if len(title) else ''
        plt = self.draw_points(X, y, title)
        plt.tight_layout()
        
        # x1 and x2 features
        x1 = X[:, 0]

        self.draw_line(plt, x1, y, c='red')
        if vertical: self.draw_vertical_line(plt, X, y)
        plt.show()

    def draw_cost_surface(self, X, y, gdpath=True):
        x1 = np.linspace(0, 40, 50, endpoint=True)
        x2 = np.linspace(-10, 30, 50, endpoint=True)

        title = 'Quadratic Cost Function Surface and Contour'
        x1, x2 = np.meshgrid(x1, x2)
        costs = np.zeros(x1.shape)
        backup_b = self.b
        backup_w1 = self.w1
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                self.b = x1[i,j], 
                self.w1 = x2[i,j]
                costs[i,j] = self.cost(X, y)

        self.b = backup_b
        self.w1 = backup_w1
        
        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x1, x2, costs, rstride=1, cstride=1, cmap='hot', 
                        edgecolor='none', alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("b")
        ax.set_ylabel("w1")
        
        if gdpath: ax.plot3D(self.bs_, self.w1s_, self.costs_, c='red')
        ax0 = plt.axes([0.1, 0.5, 0.3, 0.3])
        ax0.contour(x1, x2, costs, 20, cmap='hot')
        if gdpath: ax0.plot(self.bs_, self.w1s_, 'red')
        plt.show()

def load_linear_dataset(random_state=None, features=1, points=50):
    rng = np.random.RandomState(random_state)

    # Generate sample data
    x = 20 * rng.rand(points, features) + 2
    y = 0.5 * (x[:,0] - rng.rand(1, points)).ravel() - 1 

    return x, y

def LRTest():
    samples = 50
    X, y = load_linear_dataset(random_state=0, features=1, points=samples)

    lr = LRegress(b=15, w1=15, eta=0.1, tol=1e-4)
    X = lr.sgd(X, y, max_iter=100, standard=True)
    lr.draw_costs()
    lr.draw_separate_line(X, y)
    lr.draw_cost_surface(X, y)
    print(lr.costs_[-1])
    print(lr.predict(np.array([[0],[5],[10],[15]])))

def BMITest():
    X,y = dbload.load_bmi_dataset(standard=False)
    X = X[:,2].reshape(X.shape[0],1) # last column is BMI
    lr = LRegress(b=5, w1=5, eta=0.1, tol=0.001)
    
    X = lr.bgd(X, y, max_iter=100, standard=True)
    lr.draw_costs()
    lr.draw_separate_line(X, y)
    lr.draw_cost_surface(X, y)
    print(lr.costs_[-1])

# extend style as [x1,x2] to [1, x1, x2, x2x1, x1^2, x2^2]
def poly_extend_feature(X,degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

def BMISklearnTest():
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression(fit_intercept=False, n_jobs=None, normalize=True)
    X,y = dbload.load_bmi_dataset(standard=False)
    X = X[:,2].reshape(X.shape[0],1) # last column is BMI
    
    extend_X = poly_extend_feature(X, degree=2)
    # y = b + w1x + w2* x** 2
    lr.fit(extend_X, y)
    print(lr.coef_) 

    cost = np.sum((lr.predict(extend_X) - y)**2) / extend_X.shape[0] / 2
    print("cost:\t%f" % cost)
    print("score:\t%f" % lr.score(extend_X, y))
    print(lr.get_params())
    
    plt.figure()
    x1 = np.linspace(10, 40, 50, endpoint=True).reshape(50,1)
    extend_x1 = poly_extend_feature(x1, degree=2)
    plt.plot(x1, lr.predict(extend_x1), c='red')
    plt.scatter(X, y, c='black', marker='o')
    plt.xlabel("BMI") #'x1'
    plt.ylabel("Fat%")  #'y'    
    plt.show()

if __name__ == "__main__":
    BMISklearnTest()
