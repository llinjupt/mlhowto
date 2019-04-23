# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:44:46 2018

@author: Red
"""
    
import numpy as np
import struct
import os
import crossvalid,scaler

def __load_mnist(path, kind='train', count=0, dtype=np.uint8):
    ''' Load MNIST From idx File
    
        Parameters:
        -------------
        path: str
            dir of idx files.
        kind: 'train' or 't10k'
            train or test datase
        count: uint32
            <=0: all, number of entries
            
        Returns:
        -------------
        images: ndarray with shape (entries, 28, 28)
        labels: ndarray with shape (entries,)
    '''
    if count <= 0:
        count = -1

    abspath = os.path.abspath(path)
    labels_path = os.path.join(abspath, kind + '-labels.idx1-ubyte')
    images_path = os.path.join(abspath, kind + '-images.idx3-ubyte')

    with open(labels_path, 'rb') as flabels:
        magic, n = struct.unpack('>II', flabels.read(8))
        labels = np.fromfile(flabels, dtype=np.uint8, count=count)

    with open(images_path, 'rb') as fimages:
        magic, num, rows, cols = struct.unpack('>IIII', fimages.read(16))
        images = np.fromfile(fimages, dtype=dtype, count=count * 28 * 28)

    return images.reshape(labels.shape[0], 28, 28), labels

# count = 0 mean load all trainset
expand_file = r"./db/mnist/mnist_testexp.pkl.gz"
def expand_mnist(count=0, offset_pixels=1):
    import pickle,gzip
    import scaler
    
    ef = expand_file + str(offset_pixels)
    if os.path.exists(ef):
        print("The expanded training set already exists.")
        return
    
    x_train, y_train = __load_mnist("./db/mnist", kind='t10k', 
                                    count=count, dtype=np.uint8)
    
    # move down 1 pixel
    x_down = np.roll(x_train, offset_pixels, axis=1)
    x_down[:, 0, :] = 0
    
    # move up 1 pixel
    x_up = np.roll(x_train, -offset_pixels, axis=1)
    x_up[:, -1, :] = 0
    
    # move right 1 pixel 
    x_right = np.roll(x_train, offset_pixels, axis=2)
    x_right[:, :, 0] = 0

    # move left 1 pixel 
    x_left = np.roll(x_train, -offset_pixels, axis=2)
    x_left[:, :, -1] = 0

    expand_x = np.vstack([x_down, x_up, x_right, x_left])
    expand_x, expand_y = scaler.shuffle(expand_x, np.tile(y_train, 4))
    print("Saving expanded data. This may take a few minutes.")
    with gzip.open(ef, "w") as f:
        pickle.dump((expand_x, expand_y), f)

'''
def rotate(image, angle, scale = 1.0):
    import cv2
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))
'''

def expand_rotate_mnist(count=0, degree=10):
    import pickle,gzip
    import scaler,cv2
    
    ef = expand_file + 'rotate' + str(degree)
    if os.path.exists(ef):
        print("The expanded training set already exists.")
        return

    x_train, y_train = __load_mnist("./db/mnist", kind='t10k', 
                                    count=count, dtype=np.uint8)

    count = x_train.shape[0]
    M = cv2.getRotationMatrix2D((14,14), degree, 1)
    M1 = cv2.getRotationMatrix2D((14,14), -degree, 1)
    expand_x = np.zeros((count, 28, 28), dtype='uint8')
    expand_x1 = np.zeros((count, 28, 28), dtype='uint8')
    for i in range(count):
        expand_x[i] = cv2.warpAffine(x_train[i], M, (28, 28))
        expand_x1[i] = cv2.warpAffine(x_train[i], M1, (28, 28))
    
    expand_x = np.vstack([expand_x, expand_x1])
    expand_x, expand_y = scaler.shuffle(expand_x, np.tile(y_train, 2))
    print(expand_x.shape, expand_y.shape)
    print("Saving expanded data. This may take a few minutes.")
    with gzip.open(ef, "w") as f:
        pickle.dump((expand_x, expand_y), f)

def load_expand_mnist(count=0, offset_pixels=1):
    import pickle,gzip
    
    ef = expand_file + str(offset_pixels)
    if not os.path.exists(ef):
        print("The expanded test set not exists.")
        return

    f = gzip.open(ef, 'rb')
    expand_x, expand_y = pickle.load(f)
    f.close()
    
    count = int(count)
    if count <= 0:
       return expand_x, expand_y
    return expand_x[0:count], expand_y[0:count]

def load_expand_rotate_mnist(count=0, degree=10):
    import pickle,gzip
    
    ef = expand_file + 'rotate' + str(degree)
    if not os.path.exists(ef):
        print("The expanded test set not exists.")
        return

    f = gzip.open(ef, 'rb')
    expand_x, expand_y = pickle.load(f)
    f.close()
    
    count = int(count)
    if count <= 0:
       return expand_x, expand_y
    return expand_x[0:count], expand_y[0:count]
       
g_minst_labels = None
g_minst_images = None
g_minst_test_labels = None
g_minst_test_images = None
def load_mnist(path, kind='train', count=-1):
    '''Cached to speed loading'''
    
    global g_minst_images, g_minst_labels
    global g_minst_test_images, g_minst_test_labels
    
    count = int(count)
    if kind == 'train':
        if g_minst_labels is None or g_minst_labels.shape[0] < count:
            g_minst_images, g_minst_labels = __load_mnist(path, kind, count)
        
        if count > 0:
            return g_minst_images[0:count], g_minst_labels[0:count]
        
        return g_minst_images, g_minst_labels
    else:
        kind = 't10k'
        if g_minst_test_labels is None or g_minst_test_labels.shape[0] < count:
            g_minst_test_images, g_minst_test_labels = __load_mnist(path, kind, count)
        
        if count > 0:
            return g_minst_test_images[0:count], g_minst_test_labels[0:count]
        
        return g_minst_test_images, g_minst_test_labels

def load_mnist_vector(count=40000, test=10000):
    X_train, X_labels = load_mnist(r"./db/mnist", kind='train', count=count)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] ** 2)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-25

    ds = scaler.DataScaler(X_train)
    X_train = ds.sklearn_standard(X_train)

    y_test, y_labels = load_mnist(r"./db/mnist", kind='t10k', count=test)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1] ** 2)
    # Note: must use X_train mean and std standard testset
    y_test = (y_test - mean)/std 
    
    return X_train, X_labels, y_test, y_labels

def load_expand_mnist_vector(count=40000, test=10000):
    X_train, X_labels = load_mnist(r"./db/mnist", kind='train', count=count)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] ** 2)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-25

    ds = scaler.DataScaler(X_train)
    X_train = ds.sklearn_standard(X_train)

    y_test, y_labels = load_expand_mnist(count=test)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1] ** 2)
    # Note: must use X_train mean and std standard testset
    y_test = (y_test - mean)/std 
    
    return X_train, X_labels, y_test, y_labels

def __load_kaggele_mnist(fname, labeled=True, count=-1):
    ''' Load Kaggle Mnist From csv file
    
        Parameters:
        -------------
        fname: str
            file path of csv file.
        kind: 'train' or 't10k'
            train or test datase
        count: uint32
            -1: all, read entries
            
        Returns:
        -------------
        images: ndarray with shape (entries, 28, 28)
        labels: ndarray with shape (entries,)
    '''    
    
    if count <= 0:
        max_rows = None
    else:
        max_rows = count

    data = np.genfromtxt(fname, delimiter = ",", dtype = "uint8", 
                         skip_header=1, max_rows=max_rows)
   
    if labeled:
        target = data[:, 0]
        data = data[:, 1:].reshape(data.shape[0], 28, 28)
        return (data, target)

    return data[:, 0:].reshape(data.shape[0], 28, 28)

g_kaggle_images = None
g_kaggle_labels = None
def load_kaggele_mnist(fname, count=0):
    global g_kaggle_images, g_kaggle_labels
    
    if g_kaggle_images is None or g_kaggle_images.shape[0] < count:
        g_kaggle_images, g_kaggle_labels = __load_kaggele_mnist(fname, True, count)
    
    if count > 0:
        data = g_kaggle_images[0:count]
        target = g_kaggle_labels[0:count]
    
    ratio = 0.33
    split = int((1 - ratio) * data.shape[0])    
    return data[0:split, :, :], target[0:split]

g_kaggle_images_test = None
def load_kaggele_mnist_test(fname, count=0):
    global g_kaggle_images_test
    
    if g_kaggle_images_test is None or g_kaggle_images_test.shape[0] < count:
        g_kaggle_images_test = __load_kaggele_mnist(fname, False, count)

    if count > 0:
        data = g_kaggle_images_test[0:count]
    
    return data

def load_iris_dataset(ratio=0.3, random_state=0, negtive=-1):
    import pandas as pd
    df = pd.read_csv('db/iris/iris.data', header=0)
    
    # get the classifications
    y = df.iloc[:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, negtive)
     
    # get samples' features 2(sepal width) and 4(petal width)
    X = df.iloc[:100, [1,3]].values
    X,y = scaler.shuffle(X, y)
    X_train, X_test, y_train, y_test = crossvalid.data_split(X, y, ratio=ratio, 
                                                             random_state=random_state)
    ds = scaler.DataScaler(X_train)
    X_train = ds.sklearn_standard(X_train)
    X_test = ds.sklearn_standard(X_test)
    
    return X_train, X_test, y_train, y_test

def load_iris_mclass(ratio=0.3, random_state=0):
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data[:, [1,3]]
    y = iris.target
    X,y = scaler.shuffle(X, y)

    X_train, X_test, y_train, y_test = crossvalid.data_split(X, y, ratio=ratio, 
                                                             random_state=random_state)    
    ds = scaler.DataScaler(X_train)
    X_train = ds.sklearn_standard(X_train)
    X_test = ds.sklearn_standard(X_test)
    
    return X_train, X_test, y_train, y_test

def load_bmi_dataset(random_state=None, standard=True):
    import pandas as pd
    df = pd.read_csv('db/bmi/BMI.csv', header=0)
    
    # get the last column %Fat
    y = df.iloc[:, -1].values
     
    # Height M	Weight kg	BMI
    X = df.iloc[:, [0,1,2]].values
    if random_state is not None:
        X,y = scaler.shuffle(X, y)
    
    if not standard: return X,y
    else: return scaler.standard(X), y

# generate noraml distribution train set
def load_nd_dataset(positive=100, negtive=100, type='normal'):
    np.random.seed(3)

    if type == 'normal':
        numA = np.random.normal(4, 2, (2, positive))
        numB = np.random.normal(-4, 2, (2, negtive))
    elif type == 'ones':
        numA = np.ones((2, positive)) - 3
        numB = np.ones((2, negtive)) + 5
    else:
        numA = np.zeros((2, positive)) - 3
        numB = np.zeros((2, negtive)) + 5

    Ax, Ay = numA[0] * 0.6, numA[1]
    Bx, By = numB[0] * 1.5, numB[1]
    
    labels = np.zeros((negtive + positive, 1))
    trainset = np.zeros((negtive + positive, 2))
    trainset[0:positive,0] = Ax[:]
    trainset[0:positive,1] = Ay[:]
    labels[0:positive] = 1
    
    trainset[positive:,0] = Bx[:]
    trainset[positive:,1] = By[:]
    labels[positive:] = 0

    return trainset, labels.reshape(positive + negtive,)

if __name__ == "__main__":
    expand_rotate_mnist(degree=20)
