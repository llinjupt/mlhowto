# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:10:34 2018

@author: Red
"""

import numpy as np
import dbload
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class LogRegressGD(object):
    """Logistic Regression Classifier with gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Logistic cost function value in each epoch.
      
    """

    def __init__(self, eta=0.001, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter
        self.complex = 0 # Statistic algorithm complexity
   
    def errors(self, X, y):
        '''Statistic all errors into self.errors_'''
        predicts = self.appendedX_.dot(self.w_)
        diffs = np.where(predicts >= 0.0, self.positive, self.negtive) - y
        errors = np.count_nonzero(diffs)
        self.errors_.append(errors)
        return errors, diffs
    
    def update_labels(self, y):
        # for ploting and predict
        self.positive = np.max(y)
        self.negtive = np.min(y)
        self.positive_num = np.count_nonzero(y == self.positive)
        self.negtive_num = np.count_nonzero(y == self.negtive)

    def fit(self, X, y):
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
        self.w_ = 1.0 * np.zeros(1 + x_features) 
        self.update_labels(y)

        self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
        self.errors_ = []
        self.costs_ = []

        # record every w during whole iterations
        self.wsteps_ = []
        self.steps_ = 1  # every steps_ descent steps statistic one cose and error sample
        
        while(1):
            self.complex += 1
            
            # minmium cost function with partial derivative wi
            output = self.sigmoid(self.net_input(X))
            deltaw = (y - output)
            deltaw = self.eta * deltaw.dot(self.appendedX_)
            
            '''
            if np.max(np.abs(deltaw)) < 0.00001:
                print("deltaw is less than 0.00001")
                return self
            '''
            
            self.w_ += deltaw
            
            if(self.complex > self.n_iter):
                print("Loops beyond n_iter %d" % self.n_iter)
                return self

            if (self.complex % self.steps_ == 0):
                errors, diffs = self.errors(X, y)
                self.wsteps_.append(self.w_.copy())
                
                # compute the cost of logistic  
                diff = 1.0 - output
                diff[diff <= 0] = 1e-10
                cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(diff)))
                self.costs_.append(cost)

        return self

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
        self.w_ = 1.0 * np.zeros(1 + x_features)
        self.w_initialized = True
        self.update_labels(y)

        self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))
        self.errors_ = []
        self.costs_ = []

        # record every w during whole iterations
        self.wsteps_ = []
        self.steps_ = 1  # every steps_ descent steps statistic one cose and error sample

        while(1):
            deltaws = []
            for xi, target in zip(X, y):
                self.complex += 1
                deltaw = self.update_weights(xi, target)
                deltaws.append(deltaw)
                if (self.complex % self.steps_ == 0):
                    errors, diffs = self.errors(X, y)

                    # compute the cost of logistic  
                    output = self.sigmoid(self.net_input(X))
                    diff = 1.0 - output
                    diff[diff <= 0] = 1e-10
                    cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(diff)))
                    self.costs_.append(cost)

                    if(self.complex > self.n_iter):
                        print("Loops beyond n_iter %d" % self.n_iter)
                        return self

            '''
            if np.max(np.abs(np.array(deltaws))) < 0.0001:
                print("deltaw is less than 0.0001")
                self.wsteps_.append(self.w_.copy()) # record last w
                return self
            '''
            import scaler
            X, y = scaler.shuffle(X, y)
            self.appendedX_ = np.hstack((np.ones(samples).reshape(samples, 1), X))

        return self

    def update_weights(self, xi, target):
        '''Apply Adaline learning rule to update the weights'''

        deltaw = self.eta * (target - self.sigmoid(self.net_input(xi)))
        self.w_[1:] += xi.dot(deltaw)
        self.w_[0] += deltaw * 1
        
        return deltaw
    
    # X is a vector including features of a sample 
    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0] * 1
    
    # Activate function
    def sigmoid(self, z):
        """Compute logistic sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)

    def predict_proba(self, x):
        p = self.sigmoid(self.net_input(x))
        return np.array([p, 1-p])

    def loglikelihood_cost(self, X, y, w):
        z = np.dot(X, w[1:]) + w[0] * 1
        output = self.sigmoid(z)
        diff = 1.0 - output
        diff[diff <= 0] = 1e-15
        return -y.dot(np.log(output)) - ((1 - y).dot(np.log(diff)))

    def sse_cost(self, X, y, w):
        z = np.dot(X, w[1:]) + w[0] * 1
        sg = self.sigmoid(z)
        return np.sum((sg - y)**2) / 2

    #  MSE (Mean squared error)
    def quadratic_cost(self, X, y, w):
        #y[y == 0] = -1 # arbitray set -1 from 0 for negtive labels
        z = np.dot(X, w[1:]) + w[0] * 1
        return np.sum((y - z)**2) / X.shape[0] / 2

    def draw_quad_cost_surface(self, X, y):
        self.draw_cost_surface(X, y, cost='quad')
        
    def draw_sse_cost_surface(self, X, y):
        self.draw_cost_surface(X, y, cost='sse')

    def draw_llh_cost_surface(self, X, y):
        self.draw_cost_surface(X, y, cost='llh')

    def draw_cost_surface(self, X, y, cost='sse'):
        x1 = np.linspace(-40, 40, 90, endpoint=True)
        x2 = np.linspace(-40, 40, 90, endpoint=True)
        
        if cost == 'llh':
            cost_func = self.loglikelihood_cost
            title = 'Log-likelihood Cost Function Surface and Contour'
        elif cost=='quad':
            cost_func = self.quadratic_cost
            title = 'Quadratic Cost Function Surface and Contour'
        else:
            cost_func = self.sse_cost
            title = 'SSE Cost Function Surface and Contour'
        
        w1, w2 = np.meshgrid(x1, x2)
        costs = np.zeros(w1.shape)
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                costs[i,j] = cost_func(X, y, np.array([0, w1[i,j], w2[i,j]]))

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(w1, w2, costs, rstride=1, cstride=1, cmap='hot', edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")

        ax0 = plt.axes([0.1, 0.5, 0.3, 0.3])
        ax0.contour(w1, w2, costs, 30, cmap='hot')
        plt.show()

    def draw_errors(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Convergence errors state") # 收敛状态图
        plt.xlabel("Iterations")              # 迭代次数
        plt.ylabel("Miss-classifications")    # 误分类错误样本数

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
        plt.title("Logistic Regression vectors state")  
        
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

    def draw_points(self, X, labels, title='', figsize=(4,4), coordinate=False):
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
        
        plt.xlim(-max_, max_)
        plt.xticks(np.arange(-max_, max_))
        
        plt.ylim(-max_, max_)
        plt.yticks(np.arange(-max_, max_))

        max,min = self.positive, self.negtive
        plt.scatter(x1[labels == max], x2[labels == max], c='black', marker='o')
        plt.scatter(x1[labels == min], x2[labels == min], c='black', marker='s')
        
        if coordinate:
            for index, x, y in zip(range(len(labels)), x1, x2):
                plt.annotate('(%.2f,%.2f)'%(x,y), xy=(x,y), xytext=(-20,-20), 
                         textcoords = 'offset pixels', ha='left', va='bottom')

        return plt
    
    def draw_separate_line(self, X, y, title='', figsize=(4,4)):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        title = "Logistic Regression" + (' ' + title) if len(title) else ''
        plt = self.draw_points(X, y, title, figsize=figsize)
        
        # x1 and x2 features
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        self.draw_line(plt, self.w_, x1, x2, 0, c='black') 
        self.draw_foot_point(X,y)
        plt.show()

    def draw_converge_lines(self, X, y):
        if np.shape(self.w_)[0] != 3:
            print("can't draw the hyperplane with D%d", np.shape(self.w_)[0])
            return # can't draw the hyperplane
        
        plt = self.draw_points(X, y, "Logistic Regression convergence lines")
        
        # x1 and x2 features
        x1 = X[:, 0] 
        x2 = X[:, 1]

        self.draw_line(plt, self.w_, x1, x2, len(self.wsteps_), c='black')
        for i in range(len(self.wsteps_) - 1):
            self.draw_w_line(plt, X, i)

        plt.show()

    def draw_foot_point(self, X, y):
        import drawutils
        xaverage = X[y==self.positive].sum(axis=0) / self.positive_num
        yaverage = X[y==self.negtive].sum(axis=0) / self.negtive_num
       
        fx1, fx2 = drawutils.get_footpoint(xaverage[0], xaverage[1], self.w_)
        plt.plot([xaverage[0], fx1], [xaverage[1], fx2], c='red')
        
        plt.scatter(xaverage[0], xaverage[1], c='red', marker='p')
        plt.scatter(yaverage[0], yaverage[1], c='red', marker='p')
        
        fx1, fx2 = drawutils.get_footpoint(yaverage[0], yaverage[1], self.w_)
        plt.plot([yaverage[0], fx1], [yaverage[1], fx2], c='red')
  
def boolOrTrain():
    # Bool or Train         x11 x12 y1
    BoolOrTrain = np.array([[0, 0, -1],
                            [0, 1,  1],
                            [1, 0,  1],
                            [1, 1,  1]])
    X = BoolOrTrain[:, 0:-1]
    y = BoolOrTrain[:, -1]
    
    BoolOr = LogRegressGD(0.01, 10000)
    BoolOr.fit(X, y)

    print('Weights: %s' % BoolOr.w_)
    print('Errors: %s' % BoolOr.errors_)
    #BoolOr.draw_separate_line(X, y,'Bool-or')
    #BoolOr.draw_vectors()
    BoolOr.draw_costs()
    BoolOr.draw_converge_lines(X,y)

#irisTrainSGD(1)
def draw_normal_distribution(points=100):
    import matplotlib.pyplot as plt

    np.random.seed(1)
    rand_num = np.random.normal(0, 1, (4, points))
    Ax, Ay = rand_num[0] + 3, rand_num[1] + 3
    Bx, By = rand_num[2] - 3, rand_num[3] - 3
     
    plt.figure(figsize=(4,4))
    plt.title("Normal Distribution with {} points".format(points))
    plt.xlim(-10, 10) 
    plt.ylim(-10, 10) 

    plt.scatter(Ax, Ay, s=5, c='black', marker='o')
    plt.scatter(Bx[points // 2:], By[points // 2:], s=5, c='black', marker='o')
    plt.show()

import scaler
def normal_distribute_test():
    positive_num = 30
    negtive_num = 30
    X,y = dbload.load_nd_dataset(positive_num, negtive_num)

    X = scaler.standard(X)
    X,y = scaler.shuffle(X, y)
    ND = LogRegressGD(0.5, 1000)
    ND.fit_sgd(X, y)

    print('Weights: %s' % ND.w_)
    #print('Errors: %s' % ND.errors_[-1])
    print("LastCost: %f" % ND.costs_[-1])
    ND.draw_separate_line(X, y,'Normal Distribute')
    ND.draw_sse_cost_surface(X, y)
    #ND.draw_vectors()
    #ND.draw_costs()
    #ND.draw_converge_lines(X,y)

def irisLogRegressGD(type=0):
    X_train, X_test, y_train, y_test = dbload.load_iris_dataset(negtive=0)
    
    irisPerceptron = LogRegressGD(0.1, 1000)
    irisPerceptron.fit(X_train, y_train)

    predict = irisPerceptron.predict(X_test)
    errnum = (predict != y_test).sum()
    print("Misclassified number {}, Accuracy {:.2f}%".format(errnum, \
          (X_test.shape[0] - errnum)/ X_test.shape[0] * 100))

    irisPerceptron.draw_separate_line(X_train, y_train, 'iris')
    #irisPerceptron.draw_converge_lines(X, y)
    #irisPerceptron.draw_vectors()
    irisPerceptron.draw_costs()
    print("LastCost: %f" % irisPerceptron.costs_[-1])
    print('Weights: %s' % irisPerceptron.w_)

import drawutils
def test_plot_decision_regions():
    import dbload
    from sklearn.linear_model import LogisticRegression 
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = dbload.load_iris_mclass()
    lr = LogisticRegression(solver='lbfgs', random_state=0, multi_class='auto')
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    print("Misclassified number {}, Accuracy {:.2f}%".format((predict != y_test).sum(), 
           accuracy_score(y_test, predict)*100))

    X_all = np.vstack((X_train, X_test))
    y_all = np.hstack((y_train, y_test))
    drawutils.plot_decision_regions(X_all, y_all, clf=lr, 
                                    test_idx=range(X_train.shape[0], X_all.shape[0]))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    
    normal_distribute_test()
    