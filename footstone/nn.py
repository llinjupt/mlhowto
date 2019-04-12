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
    
    def __init__(self, sizes, eta=0.001, epochs=1000, tol=None):
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
        
        self.biases = [np.random.randn(l, 1) for l in sizes[1:]]
        self.weights = [np.random.randn(l, x) for x, l in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, X):
        out = X.T
        for b, W in zip(self.biases, self.weights):
            out = self.sigmoid(W.dot(out) + b)
        return out

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
    def mbatch_backprop(self, X, y):
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
        for b, W in zip(self.biases, self.weights):
            z = W.dot(activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            acts.append(activation)

        # backpropagation
        delta = (acts[-1] - y.T) * self.sigmoid_derivative(zs[-1])
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
        delta_w[-1] = np.dot(delta, acts[-2].transpose())
        for l in range(2, self.num_layers):
            sp = self.sigmoid_derivative(zs[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_b[-l] = np.sum(delta, axis=1, keepdims=True)
            delta_w[-l] = np.dot(delta, acts[-l-1].transpose())

        self.biases = [b-(self.eta) * nb
                        for b, nb in zip(self.biases, delta_b)]
        self.weights = [w-(self.eta) * nw 
                        for w, nw in zip(self.weights, delta_w)]
    
    def fit_mbgd(self, X, y, batchn=8, verbose=False):
        '''mini-batch stochastic gradient descent.'''
        self.errors_ = []
        self.costs_ = []
        self.steps_ = 100  # every steps_ descent steps statistic cost and error sample
        
        if batchn > X.shape[0]:
            batchn = 1
        
        for loop in range(self.epochs):
            X, y = scaler.shuffle(X, y)
            if verbose: print("Epoch: {}/{}".format(self.epochs, loop+1), flush=True)
            
            x_subs = np.array_split(X, batchn, axis=0)
            y_subs = np.array_split(y, batchn, axis=0)
            for batchX, batchy in zip(x_subs, y_subs):
                self.mbatch_backprop(batchX, batchy)

            if self.complex % self.steps_ == 0:
                cost = self.quadratic_cost(X,y)
                self.costs_.append(cost)
                if len(self.costs_) > 5:
                    if sum(self.costs_[-5:]) / 5 - self.costs_[-1] < 1e-5:
                        print("cost reduce very tiny, just quit!")
                        return

                print("costs {}".format(cost))
                if self.tol is not None and cost < self.tol:
                    print("Quit from loop for less than tol {}".format(self.tol))
                    return
            self.complex += 1
    
    # Activate function
    def sigmoid(self, z):
        """Compute logistic sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-z))

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

    def quadratic_cost(self, X, y):
        return np.sum((self.feedforward(X) - y.T)**2) / self.sizes[-1] / 2 
    
    def loglikelihood_cost(self, X, y):
        output = self.feedforward(X)

        diff = 1.0 - output
        diff[diff <= 0] = 1e-15
        return np.sum(-y.T * np.log(output) - ((1 - y.T) * np.log(diff)))       

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
        
        x1 = np.linspace(-2, 2, 80, endpoint=True)
        x2 = np.linspace(-2, 2, 80, endpoint=True)
    
        title = 'Perdict Surface and Contour'
    
        x1, x2 = np.meshgrid(x1, x2)
        acts = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                acts[i,j] = self.feedforward(np.array([x1[i,j], x2[i,j]]).reshape(1,2))

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x1, x2, acts, rstride=1, cstride=1, cmap='hot', edgecolor='none', alpha=0.8)
        
        z = self.feedforward(X)
        ax.scatter3D(X[:,0], X[:,1], z, c='blue')
        ax.scatter3D(X[:,0], X[:,1], y[:,0], c='red')
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        ax0 = plt.axes([0.1, 0.5, 0.3, 0.3])
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

def boolXorTrain():
    # Bool xor Train         x11 x12 y1
    BoolXorTrain = np.array([[1, 0,  1],
                             [0, 0,  0],
                             [0, 1,  1],
                             [1, 1,  0]])
    X = BoolXorTrain[:, 0:2]
    y = BoolXorTrain[:, 2]
    if y.ndim == 1:
        y = y.reshape(y.shape[0], 1)

    nn = NN([2,2,1], eta=0.5, epochs=10000, tol=1e-4)
    nn.fit_mbgd(X, y)
    pred = nn.predict(X)
    print("weights:", nn.weights)
    print("biases:", nn.biases)
    print(pred)
    
    nn.draw_costs()
    if nn.costs_[-1] < 1e-3:
        nn.draw_perdict_surface(X,y)

def sklearn_nn_test():
    from sklearn.neural_network import MLPClassifier

    BoolXorTrain = np.array([[0, 0,  0],
                             [1, 0,  1],
                             [0, 1,  1],
                             [1, 1,  1]])
    X_train = BoolXorTrain[:, 0:-1]
    y_train = BoolXorTrain[:, -1]

    mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=10000, alpha=1e-3, activation='logistic',
                        solver='adam', early_stopping=False, verbose=10, tol=1e-3, shuffle=True,
                        learning_rate_init=0.5)
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_train, y_train))
    print('predictions:', mlp.predict(X_train)) 
    print(mlp.predict_proba(X_train))
    
if __name__ == "__main__":
    boolXorTrain()
