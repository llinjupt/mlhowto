# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:44:46 2018

@author: Red
"""
    
import numpy as np
import struct
import os

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

g_minst_labels = None
g_minst_images = None
g_minst_test_labels = None
g_minst_test_images = None
def load_mnist(path, kind='train', count=-1):
    '''Cached to speed loading'''
    
    global g_minst_images, g_minst_labels
    global g_minst_test_images, g_minst_test_labels

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

def test():
    images, labels = load_mnist(r"./db/mnist", kind='train', count=10)
    print(images.shape)
