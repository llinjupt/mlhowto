# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:32:45 2018

@author: Red
"""

import numpy as np
import time
import dbload

def draw_normal_distribution(points=100):
    import matplotlib.pyplot as plt

    np.random.seed(0)
    rand_num = np.random.normal(0, 1, (4, points))
    Ax, Ay = rand_num[0] - 3, rand_num[1] - 3
    Bx, By = rand_num[2] + 3, rand_num[3] + 3
     
    plt.figure()
    plt.title("Normal Distribution with {} points".format(points))
    plt.xlim(-10, 10) 
    plt.ylim(-10, 10) 

    plt.scatter(Ax, Ay, s=5, c='black')
    plt.scatter(Bx, By, s=5, c='black')
    plt.show()
    
class kNN():
    def __init__(self, k=5):
        self.n_neibours = k

    def predict(self, train, labels, sample):
        import operator
        
        diff = train.astype('float64') - sample.astype('float64')
        distance = np.sum(diff ** 2, axis=2)
        distance = np.sum(distance, axis=1) ** 0.5
    
        I = np.argsort(distance)
        labels = labels[I]
        
        max_labels = {}
        k = self.n_neibours
        if len(train) < self.n_neibours:
            k = len(train)
        for i in range(0,k):
            max_labels[labels[i]] = max_labels.get(labels[i], 0) + 1
    
        return sorted(max_labels.items(), key=operator.itemgetter(1), reverse=True)

    def mnist_test(self, train_entries=10000, test_entries=10000):
        train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=train_entries)
        test,test_labels = dbload.load_mnist(r"./db/mnist", kind='test', count=test_entries)
    
        error_entries = 0
        start = time.process_time()
        for i in range(0, test_entries):
            max_labels = self.predict(train, labels, test[i])
            predict = max_labels[0][0]
            if(predict != test_labels[i]):
                error_entries += 1
                #print(predict, test_labels[i], flush=True)
                #cv2.imshow("Predict:{} Label:{}".format(predict, test_labels[i]), test[i])
    
        print("Average cost time {:.02f}ms accuracy rate {:.02f}% on trainset {}".format(
              (time.process_time() - start) / test_entries * 1000,
              (test_entries - error_entries) / test_entries * 100,
              train_entries), flush=True)
        #cv2.waitKey(0)

    def batch_test(self):
        for i in range(10000, 70000, 10000):
            self.mnist_test(i, 1000)
    
"""Implement kNN based on sklearn"""
class kNN_sklearn():
    def __init__(self, alg='auto', jobs=-1, w='distance', k=5):
        '''
        # default parameters for KNeighborsClassifier
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform')
        '''
        from sklearn.neighbors import KNeighborsClassifier
        self.knn = KNeighborsClassifier(algorithm=alg, n_jobs=jobs, 
                                        weights=w, n_neighbors=k)
        # for drawing error ratio plot
        self.train_batch = []
        self.error_ratio = []
        
    def predict(self, train, labels, test):
        self.knn.fit(train, labels)
        return self.knn.predict(test)
    
    def mnist_test(self, train_entries=10000, test_entries=1000):
        train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=train_entries)
        test,test_labels = dbload.load_mnist(r"./db/mnist", kind='test', count=test_entries)
        
        train = train.reshape((train_entries, train.shape[1] * train.shape[2]))
        test = test.reshape((test_entries, test.shape[1] * test.shape[2]))
        
        stime = time.process_time()
        wstime = time.time()
    
        predict = self.predict(train, labels, test)
        error = predict.astype(np.int32) - test_labels.astype(np.int32)
        error_entries = np.count_nonzero(error != 0)
        
        print("Average cost cpu time {:.02f}ms walltime {:.02f}s"
              " accuracy rate {:.02f}% on trainset {}".format(
              (time.process_time() - stime) / test_entries * 1000,
              (time.time() - wstime),
              (test_entries - error_entries) / test_entries * 100,
              train_entries), flush=True)
        
        self.train_batch.append(train_entries)
        self.error_ratio.append(error_entries / test_entries * 100)

    def batch_test(self, start=1000, step=5000):
        for i in range(start, 60000, step):
            self.mnist_test(i, 10000)

    def error_ratio_draw(self):
        '''Draw errors info with matplotlib'''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Error ratio status")
        plt.xlabel("Train dataset")         
        plt.ylabel("Error ratio(%)")        
        
        plt.plot(self.train_batch, self.error_ratio, c='grey')
        plt.scatter(self.train_batch, self.error_ratio, c='black')
        plt.show()

if __name__ == '__main__':
    # knn = kNN()
    # knn.batch_test()
    knn_sk = kNN_sklearn(alg='brute')
    knn_sk.batch_test()
    knn_sk.error_ratio_draw()
    