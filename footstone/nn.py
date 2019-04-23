# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:10:34 2018

@author: Red
"""

import numpy as np
import dbload
import scaler
import matplotlib.pyplot as plt

class NN(object):
    """Neural Network.

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
    
    """
    
    def __init__(self, sizes, eta=0.001, epochs=1000, tol=None, alpha=0, softmax=True):
        '''
        Parameters
        ------------
        eta : float
          Learning rate (between 0.0 and 1.0)
        epochs: uint
          Training epochs
        sizes : array like [3,2,3]
          Passes the layers.
        '''
        self.eta = eta
        self.epochs = epochs
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.complex = 0
        self.tol = tol
        self.alpha = alpha # for regulization
        
        self.outactive = self.sigmoid
        if softmax: self.outactive = self.softmax
        
        np.random.seed(None)
        
        if 0:
            self.biases = [np.random.randn(l, 1) for l in sizes[1:]]
            self.weights = [np.random.randn(l, x) for x, l in zip(sizes[:-1], sizes[1:])]
        else:
            std = 1/np.power(sizes[0], 0.5)
            self.biases = [np.random.normal(0, std, (l, 1)) for l in sizes[1:]]
            self.weights = [np.random.normal(0, std, (l, x)) for x, l in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, X):
        out = X.T
        for b, W in zip(self.biases[0:-1], self.weights[0:-1]):
            out = self.sigmoid(W.dot(out) + b)
        return self.outactive(self.weights[-1].dot(out) + self.biases[-1])

    # x is a sample vector with n dim
    def backprop(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)
        activation = x
        acts = [x] # list for all activations layer by layer
        zs = []           # z vectors layer by layer
        for b, W in zip(self.biases, self.weights):
            z = W.dot(activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            acts.append(activation)

        # backpropagation
        delta = (acts[-1] - y) * self.sigmoid_derivative(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, acts[-2].transpose())
        for l in range(2, self.num_layers):
            sp = self.sigmoid_derivative(zs[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, acts[-l-1].transpose())
        return delta_b, delta_w

    def mbatch_train(self, X, y):
        '''train using backpropagation on a minibatch'''
        delta_b_all = [np.zeros(b.shape) for b in self.biases]
        delta_w_all = [np.zeros(w.shape) for w in self.weights]
        
        for Xi, yi in zip(X, y):
            delta_b, delta_w = self.backprop(Xi, yi)
            delta_b_all = [nb+dnb for nb, dnb in zip(delta_b, delta_b_all)]
            delta_w_all = [nw+dnw for nw, dnw in zip(delta_w, delta_w_all)]

        self.biases = [b-(self.eta) * nb / X.shape[0]
                        for b, nb in zip(self.biases, delta_b_all)]
        self.weights = [w-(self.eta) * nw / X.shape[0]
                        for w, nw in zip(self.weights, delta_w_all)]
    
    # X is an array with n * m, n samples and m features every sample
    def mbatch_backprop(self, X, y, type='llh', total=1):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        if X.ndim == 1:
            X = X.reshape(1, X.ndim)
        if y.ndim == 1:
            y = y.reshape(1, y.ndim)
        
        activation = X.T
        acts = [activation] # list for all activations layer by layer
        zs = []             # z vectors layer by layer
        
        layers = 0
        for b, W in zip(self.biases, self.weights):
            z = W.dot(activation) + b
            zs.append(z)
            if layers == self.num_layers - 2:
                activation = self.outactive(z)
            else:
                activation = self.sigmoid(z)

            acts.append(activation)
            layers += 1

        # backpropagation
        samples = X.shape[0]
        if type == 'llh':
            delta = (acts[-1] - y.T)
        else:
            delta = (acts[-1] - y.T) * self.sigmoid_derivative(zs[-1])
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True) 
        delta_w[-1] = np.dot(delta, acts[-2].transpose())  
        for l in range(2, self.num_layers):
            sp = self.sigmoid_derivative(zs[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_b[-l] = np.sum(delta, axis=1, keepdims=True) 
            delta_w[-l] = np.dot(delta, acts[-l-1].transpose())
        
        self.biases = [b - self.eta*nb  / samples
                        for b, nb in zip(self.biases, delta_b)]

        # L2 regulization
        self.weights = [(1-self.eta*self.alpha/total)*w - self.eta * nw / samples
                        for w, nw in zip(self.weights, delta_w)]
        
        self.delta_bs_.append(delta_b)
        self.delta_ws_.append(delta_w)
        '''
        # L1 regulization
        self.weights = [-self.alpha / total - self.eta * nw 
                        for w, nw in zip(self.weights, delta_w)]
        '''
        '''
        # no regulization
        self.weights = [w-(self.eta) * nw / samples
                        for w, nw in zip(self.weights, delta_w)]
        '''
        for i in range(len(delta_b)):
            delta_b[i] /= samples
        for i in range(len(delta_w)):
            delta_w[i] /= samples
        
        return delta_b, delta_w

    def evaluate(self, x_train, x_labels, y_test, y_labels):
        pred = self.predict(x_train)
        error = pred - x_labels
        error_entries = np.count_nonzero(error != 0)

        test_entries = x_labels.shape[0]
        self.trains_.append((test_entries - error_entries) / test_entries * 100)
        print("Accuracy rate {:.02f}% on trainset {}".format(
              self.trains_[-1], test_entries), flush=True)
        
        pred = self.predict(y_test)
        error = pred - y_labels
        error_entries = np.count_nonzero(error != 0)
        test_entries = y_labels.shape[0]
        self.tests_.append((test_entries - error_entries) / test_entries * 100)
        print("Accuracy rate {:.02f}% on validateset {}".format(
              self.tests_[-1], test_entries), flush=True)
        
    def gd_checking(self, X, y, costtype='llh'): 
        # init all weights as 1
        self.biases = [np.ones((l, 1)) for l in self.sizes[1:]]
        self.weights = [np.ones((l, x)) for x, l in zip(self.sizes[:-1], self.sizes[1:])]
        epsilon = 1e-4
        
        costfunc = self.quadratic_cost
        if costtype == 'llh':
            costfunc = self.loglikelihood_cost
        
        partial_biases = [np.zeros((l, 1)) for l in self.sizes[1:]]
        for i in range(len(self.biases)):
            for j in range(self.biases[i].size):
                self.biases[i][j] += epsilon
                plus = costfunc(X,y)
                self.biases[i][j] -= 2*epsilon
                minus = costfunc(X,y)
                self.biases[i][j] += epsilon
                partial_biases[i][j] = ((plus - minus)/ 2 / epsilon)
        
        partial_weights = [np.zeros((l, x)) for x, l in zip(self.sizes[:-1], self.sizes[1:])]
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):    
                    self.weights[i][j,k] += epsilon
                    plus = costfunc(X,y)
                    self.weights[i][j,k] -= 2*epsilon
                    minus = costfunc(X,y)
                    self.weights[i][j,k] += epsilon
                    partial_weights[i][j,k] = ((plus - minus)/ 2 / epsilon)
        
        delta_b, delta_w = self.mbatch_backprop(X,y, type=costtype)
        
        # checking gradient here
        diff_bs = [b - nb for b, nb in zip(partial_biases, delta_b)]
        diff_ws = [w - nw for w, nw in zip(partial_weights, delta_w)]
        
        print(diff_bs) # must be tiny vlaues
        print(diff_ws)

    def fit_mbgd(self, X_train, y_train, batchn=8, verbose=False, costtype='llh', 
                 x_labels=None, y_test=None, y_labels=None):
        '''mini-batch stochastic gradient descent.'''
        self.trains_ = []
        self.tests_ = []
        self.errors_ = []
        self.costs_ = []
        self.steps_ = 1  # every steps_ descent steps statistic cost and error sample
        
        self.delta_bs_ = []
        self.delta_ws_ = []
        
        total = X_train.shape[0]
        if batchn > total:
            batchn = 1
            
        self.eta_reduces = 1 # accuracy prompt very tiny then try little eta
        
        for loop in range(self.epochs):
            X, y = scaler.shuffle(X_train, y_train)
            if verbose:print("Epoch: {}/{}".format(self.epochs, loop+1), flush=True)

            x_subs = np.array_split(X, batchn, axis=0)
            y_subs = np.array_split(y, batchn, axis=0)
            for batchX, batchy in zip(x_subs, y_subs):
                self.mbatch_backprop(batchX, batchy, type=costtype, total=total)

            if self.complex % self.steps_ == 0:
                if costtype == 'llh':
                    cost = self.loglikelihood_cost(X,y)
                else:
                    cost = self.quadratic_cost(X,y)
                
                self.costs_.append(cost)
                if self.tol is not None and len(self.costs_) > 5:
                    if abs(sum(self.costs_[-5:]) / 5 - self.costs_[-1]) < self.tol * self.steps_:
                        print("cost reduce very tiny less than tol, just quit!")
                        return
                
                if cost < 1e-2:
                    print("cost is very tiny just quit!")
                    return
                
                print("costs {}".format(cost))
                if y_test is not None:
                    self.evaluate(X_train,x_labels,y_test,y_labels)
                    if len(self.tests_) > 3:
                        if abs(sum(self.tests_[-3:]) / 3 - self.tests_[-1]) < 1e-3:
                            print("test evalution prompt very tiny, just quit!")
                            self.eta /= np.power(2,self.eta_reduces)
                            self.eta_reduces += 1
                            if (self.eta_reduces > 5):
                                print("reduces eta three times just quit")
                                return

            self.complex += 1
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0) 

    # Activate function
    def sigmoid(self, z):
        """Compute logistic sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def sigmoid_derivative(self, z):
        sg = self.sigmoid(z)
        return sg * (1.0 - sg)

    def predict(self, X):
        ff = self.feedforward(X)
        if self.sizes[-1] == 1: # 1 output node
            return np.where(ff >= 0.5, 1, 0)

        return np.argmax(ff, axis=0)

    def predict_proba(self, x):
        p = self.sigmoid(self.net_input(x))
        return np.array([p, 1-p])

    def regulization_cost(self, X, y):
        return 0.5 * (self.alpha / y.shape[0]) * \
               sum(np.linalg.norm(w)**2 for w in self.weights)

    def quadratic_cost(self, X, y):
        cost = np.sum((self.feedforward(X) - y.T)**2) / y.shape[0] / 2 
        if self.alpha:
            cost += self.regulization_cost(X,y)
        return cost
    
    def loglikelihood_cost(self, X, y):
        output = self.feedforward(X)
        
        diff = 1.0 - output
        diff[diff <= 0] = 1e-15
        cost = np.sum(-y.T * np.log(output) - ((1 - y.T) * np.log(diff))) / y.shape[0]
        
        if self.alpha:
            cost += self.regulization_cost(X,y)
        return cost

    def draw_quad_cost_surface(self, X, y):
        self.draw_cost_surface(X, y, cost='quad')

    def draw_llh_cost_surface(self, X, y):
        self.draw_cost_surface(X, y, cost='llh')

    def draw_cost_surface(self, X, y, cost='quad'):
        from mpl_toolkits import mplot3d

        x1 = np.linspace(-10, 10, 80, endpoint=True)
        x2 = np.linspace(-10, 10, 80, endpoint=True)

        cost_func = self.quadratic_cost
        title = 'Quadratic Cost Surface and Contour'

        if cost == 'llh':
            cost_func = self.loglikelihood_cost
            title = 'Log-likelihood Cost Surface and Contour'

        backup_w1 = self.weights[0][0,0]
        backup_w2 = self.weights[1][0,1]
        
        w1, w2 = np.meshgrid(x1, x2)
        costs = np.zeros(w1.shape)
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                self.weights[0][0,0] = w1[i,j]
                self.weights[1][0,1] = w2[i,j]
                costs[i,j] = cost_func(X, y)

        self.weights[0][0,0] = backup_w1
        self.weights[0][0,1] = backup_w2

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(w1, w2, costs, rstride=1, cstride=1, cmap='hot', edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")

        ax0 = plt.axes([0.1, 0.5, 0.3, 0.3])
        ax0.contour(w1, w2, costs, 30, cmap='hot')
        plt.show()

    def draw_perdict_surface(self, X, y):
        from mpl_toolkits import mplot3d
        
        x1 = np.linspace(-4, 4, 80, endpoint=True)
        x2 = np.linspace(-4, 4, 80, endpoint=True)
    
        title = 'Perdict Surface and Contour'
    
        x1, x2 = np.meshgrid(x1, x2)
        acts = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                acts[i,j] = self.feedforward(np.array([x1[i,j], x2[i,j]]).reshape(1,2))

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x1, x2, acts, rstride=1, cstride=1, cmap='hot', edgecolor='none', alpha=0.8)
        
        positive = (y[:,0] == 1)
        negtive = (y[:,0] == 0)
        ax.scatter3D(X[positive,0], X[positive,1], y[positive,0], c='red', marker='o')
        ax.scatter3D(X[negtive,0], X[negtive,1], y[negtive,0], c='black', marker='x')
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        ax0 = plt.axes([0.1, 0.5, 0.3, 0.3])
        
        markers = ('x', 'o', 's', 'v')
        colors = ('black', 'red', 'cyan', 'blue')
        y = y.ravel()
        for idx, cl in enumerate(np.unique(y)):
            ax0.scatter(X[y == cl, 0], X[y == cl, 1], alpha=0.8, c=colors[idx],
                        marker=markers[idx], label=cl, s=5)
        
        #ax0.scatter(X[:,0], X[:,1], c='black', s=5)
        ax0.contour(x1, x2, acts, 30, cmap='hot')
        plt.show()

    def draw_costs(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        if len(self.costs_) <= 1:
            print("can't plot costs for less data")
            return
        
        plt.figure()
        plt.title("Cost values J(w) state")   # 收敛状态图
        plt.xlabel("Iterations")              # 迭代次数
        plt.ylabel("Cost values J(w)")        # 代价函数值
        
        x = np.arange(1, 1 + len(self.costs_), 1) * self.steps_
        plt.xlim(1 * self.steps_, len(self.costs_) * self.steps_) 
        plt.ylim(0, max(self.costs_) + 1)
        #plt.yticks(np.linspace(0, max(self.costs_) + 1, num=10, endpoint=True))
        
        plt.plot(x, self.costs_, c='grey')
        plt.scatter(x, self.costs_, c='black')
        plt.show()

    def draw_evaluate(self):
        '''Draw evaluate curve matplotlib'''
        import matplotlib.pyplot as plt

        if len(self.trains_) <= 1:
            print("can't plot trains without data")
            return
        
        plt.figure()
        plt.title("Trainset and Validation set evaluation state")  
        plt.xlabel("Iterations")                            
        plt.ylabel("Accuracy Percent (%)")  
        
        plt.xlim(1 * self.steps_, len(self.trains_) * self.steps_) 
        plt.ylim(max(min(self.tests_) - 10, 0), max(self.trains_) + 1)
        
        x = np.arange(1, 1 + len(self.trains_), 1) * self.steps_
        plt.plot(x, self.trains_, c='blue', label='Train set')
        plt.scatter(x, self.trains_, c='blue')

        plt.plot(x, self.tests_, c='red', label='Validation set')
        plt.scatter(x, self.tests_, c='red')
        plt.legend(loc='upper left')
        plt.show()

    def draw_delta_b(self):
        bs = {}
        for entry in self.delta_bs_:
            for l in range(len(entry)):
                for i in range(entry[l].size):
                    key = '$b^{(' + str(l+2)+ ')}_' + str(i+1) + '$'
                    try:
                        bs[key].append(entry[l][i,0])
                    except:
                        bs[key] = []
                        bs[key].append(entry[l][i,0])

        plt.figure()
        plt.title("Delta biases status")  
        plt.xlabel("Iterations")                            
        plt.ylabel("$\Delta b$")
        for key in bs:
            plt.plot(bs[key], label=key)
        plt.legend(loc='upper left')
        plt.show()

    def draw_delta_w(self):
        ws = {}
        for entry in self.delta_ws_:
            for l in range(len(entry)):
                for i in range(entry[l].shape[0]):
                    for j in range(entry[l].shape[1]):
                        key = '$w^{(' + str(l+1)+ ')}_{' + str(i+1) + str(j+1) + '}$'
                        try:
                            ws[key].append(entry[l][i,j])
                        except:
                            ws[key] = []
                            ws[key].append(entry[l][i,j])

        plt.figure()
        plt.title("Delta weights status")  
        plt.xlabel("Iterations")                            
        plt.ylabel("$\Delta w$")
        for key in ws:
            plt.plot(ws[key], label=key)
        plt.legend(loc='upper left')
        plt.show()
        
def boolXorTrain():
    # Bool xor Train         x11 x12 y1
    BoolXorTrain = np.array([[-1, -1, 1],
                             [-2, -2, 0],
                             [1, 0,  1],
                             [0, 0,  0],
                             [0, 1,  1],
                             [1, 1,  0],
                             [2, 2,  1],
                             [3, 3,  0]])
    X = BoolXorTrain[:, 0:2]
    y = BoolXorTrain[:, 2]
    if y.ndim == 1:
        y = y.reshape(y.shape[0], 1)

    nn = NN([2,10,1], eta=0.5, epochs=10000, tol=0, alpha=0)

    nn.fit_mbgd(X, y, costtype='llh')
    pred = nn.predict(X)
    print("weights:", nn.weights)
    print("biases:", nn.biases)
    print(pred)

    nn.draw_costs()
    #nn.draw_delta_b()
    if nn.costs_[-1] < 1e-1:
        nn.draw_perdict_surface(X,y)
        #nn.draw_llh_cost_surface(X,y)
        #nn.draw_quad_cost_surface(X,y)

def irisTrain():
    X_train, X_test, y_train, y_test = dbload.load_iris_dataset(negtive=0) 
    if y_train.ndim == 1:
        y_train = y_train.reshape(y_train.shape[0], 1)
    
    nn = NN([2,4,1], eta=1, epochs=100000, tol=1e-4)
    nn.fit_mbgd(X_train, y_train, costtype='llh')

    print("weights:", nn.weights)
    print("biases:", nn.biases)

    pred = nn.predict(X_test)
    print(pred)
    error = pred - y_test
    error_entries = np.count_nonzero(error != 0)

    test_entries = y_test.shape[0]
    print("Accuracy rate {:.02f}% on trainset {}".format(
          (test_entries - error_entries) / test_entries * 100,
          test_entries), flush=True)

    nn.draw_costs()
    if nn.costs_[-1] < 1e-2:
        nn.draw_perdict_surface(X_train, y_train)
        nn.draw_perdict_surface(X_test, y_test.reshape(y_test.shape[0], 1))

def holdout_score(clf, X_test, y_test):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    pred = clf.predict(X_test)
    error = pred - y_test
    error_entries = np.count_nonzero(error != 0)

    test_entries = y_test.shape[0]
    return (test_entries - error_entries) / test_entries * 100

def kfold_estimate(k=10, type='scv'):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    images, labels, y_test, y_labels = dbload.load_mnist_vector(count=40000, test=10000)

    scores_train = []
    scores_validate = []
    scores_test = []
    if type == 'cv':
        cv = KFold(n_splits=k, random_state=1)
    else:
        cv = StratifiedKFold(n_splits=k, random_state=1)

    for train, test in cv.split(images, labels):
        X_images, X_labels = images[train], labels[train]
        y = np.zeros((X_labels.shape[0], 10))
        for i, j in enumerate(X_labels):
            y[i,j] = 1

        nn = NN([X_images.shape[1], 50, 10], eta=9, epochs=100, tol=0.01)
        nn.fit_mbgd(X_images, y, costtype='llh', batchn=64)
        
        score = holdout_score(nn, X_images, X_labels)        
        test_entries = X_labels.shape[0]
        print("Accuracy rate {:.02f}% on trainset {}".format(
              score, test_entries), flush=True)
        scores_train.append(score)
        
        score = holdout_score(nn, images[test], labels[test])        
        test_entries = labels[test].shape[0]
        print("Accuracy rate {:.02f}% on vcset {}".format(
              score, test_entries), flush=True)
        scores_validate.append(score)
        
        score = holdout_score(nn, y_test, y_labels)        
        test_entries = y_test.shape[0]
        print("Accuracy rate {:.02f}% on testset {}".format(
              score, test_entries), flush=True)
        scores_test.append(score)

    print(scores_train, "%.3f +/- %.3f" %(np.mean(scores_train), np.std(scores_train)))
    print(scores_validate,"%.3f +/- %.3f" %(np.mean(scores_validate), np.std(scores_validate)))
    print(scores_test, "%.3f +/- %.3f" %(np.mean(scores_test), np.std(scores_test)))

def MNISTTrain():
    import crossvalid
    images, labels, y_test, y_labels = dbload.load_mnist_vector(count=10000, test=100)
    X_images, validate, X_labels, validate_labels = \
        crossvalid.data_split(images, labels, ratio=0.1, random_state=None)

    y = np.zeros((X_labels.shape[0], 10))
    for i, j in enumerate(X_labels):
        y[i,j] = 1

    nn = NN([X_images.shape[1], 100, 10], eta=1, epochs=100000, tol=1e-6, alpha=15)
    nn.fit_mbgd(X_images, y, costtype='llh', batchn=256, x_labels=X_labels, 
                y_test=validate, y_labels=validate_labels)

    pred = nn.predict(y_test)
    error = pred - y_labels
    error_entries = np.count_nonzero(error != 0)

    test_entries = y_test.shape[0]
    print("Accuracy rate {:.02f}% on trainset {}".format(
          (test_entries - error_entries) / test_entries * 100,
          test_entries), flush=True)
    nn.draw_evaluate()
    nn.draw_costs()
    
def draw_perdict_surface(proba, X, y):
    from mpl_toolkits import mplot3d

    if y.ndim == 1:
        y = y.reshape(y.shape[0], 1)
    
    x1 = np.linspace(-4, 4, 80, endpoint=True)
    x2 = np.linspace(-4, 4, 80, endpoint=True)

    title = 'Perdict Surface and Contour'

    x1, x2 = np.meshgrid(x1, x2)
    acts = np.zeros(x1.shape)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            acts[i,j] = proba(np.array([x1[i,j], x2[i,j]]).reshape(1,2))[:,1]
            
    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x1, x2, acts, rstride=1, cstride=1, cmap='hot', edgecolor='none', alpha=0.8)
    
    positive = (y[:,0] == 1)
    negtive = (y[:,0] == 0)
    ax.scatter3D(X[positive,0], X[positive,1], y[positive,0]+0.1, c='red', marker='o')
    ax.scatter3D(X[negtive,0], X[negtive,1], y[negtive,0]+0.02, c='black', marker='x')
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax0 = plt.axes([0.1, 0.5, 0.3, 0.3])
    
    markers = ('x', 'o', 's', 'v')
    colors = ('black', 'red', 'cyan', 'blue')
    y = y.ravel()
    for idx, cl in enumerate(np.unique(y)):
        ax0.scatter(X[y == cl, 0], X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, s=5)
    
    #ax0.scatter(X[:,0], X[:,1], c='black', s=5)
    ax0.contour(x1, x2, acts, 30, cmap='hot')
    plt.show()

def draw_loss(loss):
    '''Draw errors info with matplotlib'''
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Loss values J(w) state") 
    plt.xlabel("Iterations")          
    plt.ylabel("Loss values J(w)")   
    
    size = len(loss)
    x = np.arange(1, 1 + size, 1) 
    plt.xlim(1, size + 1) 
    plt.ylim(0, np.max(loss) + 1)
    #plt.yticks(np.linspace(0, max(self.costs_) + 1, num=10, endpoint=True))
    
    plt.plot(x, loss, c='grey')
    plt.scatter(x, loss, c='black')
    plt.show()

def sklearn_XorTrain():
    # Bool xor Train         x11 x12 y1
    BoolXorTrain = np.array([[-1, -1, 1],
                             [-2, -2, 0],
                             [1, 0,  1],
                             [0, 0,  0],
                             [0, 1,  1],
                             [1, 1,  0],
                             [2, 2,  1],
                             [3, 3,  0]])
    X = BoolXorTrain[:, 0:2]
    y = BoolXorTrain[:, 2]

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=100000, activation='logistic',
                        solver='sgd', early_stopping=False, verbose=1, tol=1e-8, shuffle=True,
                        learning_rate_init=0.01)
    mlp.fit(X, y)
    draw_loss(mlp.loss_curve_)
    draw_perdict_surface(mlp.predict_proba, X, y)
    print(mlp.predict(X))

def sklearn_mnist(ratio=1, hidden_neurons=40, alpha=0):
    from sklearn.neural_network import MLPClassifier
    images, labels, y_test, y_labels = dbload.load_mnist_vector(
            count=int(40000 * ratio), test=int(10000 * ratio))
   
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_neurons,), max_iter=10000, activation='relu',
                        solver='lbfgs', early_stopping=False, verbose=1, tol=1e-6, shuffle=True,
                        learning_rate_init=0.01, alpha=alpha)

    mlp.fit(images, labels)
    print(mlp.score(images, labels), mlp.score(y_test, y_labels))
    return mlp.score(images, labels), mlp.score(y_test, y_labels)

def plot_learning_curve():
    train_scores,test_scores,percents = [],[],[]
    for i in np.linspace(1, 20, 50, endpoint=True):
        print("round %f" % i)
        train_score, test_score = sklearn_mnist(i * 0.1, hidden_neurons=100, alpha=0.1)
        print("Training set score: %f" % train_score)
        print("Test set score: %f" % test_score)
        train_scores.append(train_score * 100)
        test_scores.append(test_score * 100)
        percents.append(i*0.1*40000)

    plt.figure()
    plt.title("Train and Test scores status") 
    plt.xlabel("Trainset samples") 
    plt.ylabel("Scores")

    plt.plot(percents, train_scores, label='Train Scores', c='black')
    plt.plot(percents, test_scores, label='Test Scores', c='gray')
    plt.scatter(percents, train_scores, c='black')
    plt.scatter(percents, test_scores, c='gray')
    
    plt.ylim(min(test_scores) - 1, 100)
    # target horizontal line
    #
    plt.hlines((train_scores[-1] + test_scores[-1]) / 2, 0, 40000, alpha=0.5, linestyle='--')
    plt.legend(loc='upper right')
    plt.show()

def plot_validation_curve():
    train_scores,test_scores,ks = [],[],[]

    for i in np.linspace(2, 40, 30, endpoint=True):
        print("round %f" % i)
        train_score, test_score = sklearn_mnist(1, hidden_neurons=int(i), alpha=0)
        print("Training set score: %f" % train_score)
        print("Test set score: %f" % test_score)
        train_scores.append(train_score * 100)
        test_scores.append(test_score * 100)
        ks.append(i)

    plt.figure()
    plt.title("Train and Test scores status") 
    plt.xlabel("Hidden Neurons") 
    plt.ylabel("Scores")

    plt.plot(ks, train_scores, label='Train Scores', c='black')
    plt.plot(ks, test_scores, label='Test Scores', c='gray')
    plt.scatter(ks, train_scores, c='black')
    plt.scatter(ks, test_scores, c='gray')
    
    plt.ylim(min(test_scores) - 1, 100)
    plt.legend(loc='lower left')
    plt.show()

def sklearn_nn_test():
    from sklearn.neural_network import MLPClassifier
    images, labels, y_test, y_labels = dbload.load_mnist_vector(count=40000, test=10000)

    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10000, activation='relu',
                        solver='adam', early_stopping=False, verbose=1, tol=1e-6, shuffle=True,
                        learning_rate_init=0.01, alpha=0.5)

    mlp.fit(images, labels)
    print("Training set score: %f" % mlp.score(images, labels))
    print("Test set score: %f" % mlp.score(y_test, y_labels))
    #print('predictions:', mlp.predict(X_train)) 
    #print(mlp.predict_proba(X_train))

def pipline_test():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('scl', StandardScaler()),
                     ('lr', LinearRegression(fit_intercept=True))])
    X = np.linspace(1,11,10).reshape(10,1)
    y = X * 2 + 1
    pipe.fit(X, y)
    print(pipe.predict([[1],[2]]))        

if __name__ == "__main__":
    sklearn_mnist(1)
