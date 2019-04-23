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
    
    # 'x_train, x_test, y_train, y_test = '
    return train_test_split(X, y, test_size=ratio, random_state=random_state)

# extend style as [x1,x2] to [1, x1, x2, x2x1, x1^2, x2^2]
def data_extend_feature(X, degree=2, interaction_only=False, bias=True):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                              include_bias=bias)
    return poly.fit_transform(X)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
def sklearn_learning_curve():
    from sklearn.model_selection import learning_curve    
    mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, activation='relu',
                        solver='lbfgs', early_stopping=False, verbose=1, tol=1e-6, shuffle=True,
                        learning_rate_init=0.001, alpha=0)
    
    pipelr = Pipeline([('scl', StandardScaler()),
                       ('clf', mlp)])

    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target

    train_sizes, train_scores, valid_scores = \
        learning_curve(estimator=pipelr, X=X_train,y=y_train,
                       train_sizes=np.linspace(0.2, 1, 10, endpoint=True),
                       cv=5, n_jobs=8)
    train_mean = np.mean(train_scores * 100, axis=1)
    train_std = np.std(train_scores * 100, axis=1)
    valid_mean = np.mean(valid_scores * 100, axis=1) 
    valid_std = np.std(valid_scores * 100, axis=1)

    plt.title('IRIS Data Learning Curve')
    plt.plot(train_sizes, train_mean, color='black', marker='o', label='Train Scores')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, 
                     alpha=0.2, color='black')
    plt.plot(train_sizes, valid_mean, color='purple', marker='s', label='Validation Scores')
    plt.fill_between(train_sizes, valid_mean + valid_std, valid_mean - valid_std, 
                     alpha=0.15, color='purple')
    plt.xlabel('Trainset samples')
    plt.ylabel('Scores')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def sklearn_validation_curve():
    from sklearn.model_selection import validation_curve
    mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, activation='relu',
                        solver='lbfgs', early_stopping=False, verbose=1, tol=1e-6, shuffle=True,
                        learning_rate_init=0.001)
    
    pipelr = Pipeline([('scl', StandardScaler()),
                       ('clf', mlp)])

    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    
    param_range = (0, 0.001, 0.01, 0.1, 1.0, 10)
    train_scores, valid_scores = \
        validation_curve(estimator=pipelr, X=X_train,y=y_train,
                         param_name='clf__alpha', param_range = param_range,
                         cv=5, n_jobs=8)
    train_mean = np.mean(train_scores * 100, axis=1)
    train_std = np.std(train_scores * 100, axis=1)
    valid_mean = np.mean(valid_scores * 100, axis=1) 
    valid_std = np.std(valid_scores * 100, axis=1)

    plt.title('IRIS Data Validation Curve')
    plt.plot(param_range, train_mean, color='black', marker='o', label='Train Scores')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, 
                     alpha=0.2, color='black')
    plt.plot(param_range, valid_mean, color='purple', marker='s', label='Validation Scores')
    plt.fill_between(param_range, valid_mean + valid_std, valid_mean - valid_std, 
                     alpha=0.15, color='purple')
    plt.xlabel('Regulization Parameter \'alpha\'')
    plt.ylabel('Scores')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim([80,100])
    plt.xscale('log')
    plt.show()

def sklearn_grid_search():
    from sklearn.grid_search import GridSearchCV
    mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, activation='relu',
                        solver='lbfgs', early_stopping=False, verbose=1, tol=1e-6, shuffle=True,
                        learning_rate_init=0.001)
    
    pipelr = Pipeline([('scl', StandardScaler()),
                       ('clf', mlp)])

    iris = datasets.load_iris()
    X_train = iris.data
    X_labels = iris.target
    
    alpha_range = (0, 0.001, 0.01, 0.1, 1.0, 10)
    tol_range = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    act_range = ('relu', 'logistic')
    solver_range = ('lbfgs', 'sgd', 'adam')
    param_grid = {'clf__alpha': alpha_range,
                  'clf__tol': tol_range,
                  'clf__activation': act_range,
                  'clf__solver': solver_range}
    gsclf = GridSearchCV(estimator=pipelr,
                         param_grid=param_grid,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=8)
    gsclf.fit(X_train, X_labels)
    print(gsclf.best_score_)
    print(gsclf.best_params_)
    
    bestclf = gsclf.best_estimator_
    bestclf.fit(X_train, X_labels)
    
    return bestclf

def sklearn_random_search():
    from time import time
    from scipy.stats import randint as sp_randint
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    
    # get some data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # build a classifier
    clf = RandomForestClassifier(n_estimators=20)

    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
    
    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

if __name__ == "__main__":
    X,y = normal_dis_trainset(3, 3)
    
    X_train, X_test, y_train, y_test = data_split(X, y)
    print(y_train)
    print(y_test)
