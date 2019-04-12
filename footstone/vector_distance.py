# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:32:45 2018

@author: Red
"""

import numpy as np
import cv2

def bitwise(imga, imgb=None, opt='not'):
    '''bitwise: and or xor and not'''
    if opt != 'not' and imga.shape != imgb.shape:
        print("Imgs with different shape, can't do bitwise!")
        return None

    opt = opt.lower()[0]
    if opt == 'a':
        return cv2.bitwise_and(imga, imgb)
    elif opt == 'o':        
        return cv2.bitwise_or(imga, imgb)
    elif opt == 'x':
        return cv2.bitwise_xor(imga, imgb)
    elif opt == 'n':
        return cv2.bitwise_not(imga)

    print("Unknown bitwise opt %s!" % opt)
    return None

def vector_dist(V0, V1):
    from numpy import linalg as la
    V0 = np.array(V0).astype('float64')
    V1 = np.array(V1).astype('float64')

    return la.norm(V1 - V0)

def show_simple_distance():
    gray0 = np.array([[0,0],[255,255]], dtype=np.uint8)
    gray1 = gray0.transpose()
    
    cv2.imshow('-', gray0)
    cv2.imshow('|', gray1)
    
    gray2 = bitwise(gray0, None, opt='not')
    cv2.imshow('_', gray2)
    
    g01 = vector_dist(gray0, gray1)
    g02 = vector_dist(gray0, gray2)
    print("distance between -| is {}, distance between -_ is {}".format(int(g01), int(g02)))
    
    cv2.waitKey(0)

# imgs with shape(count,height,width)
def show_gray_imgs(imgs, title=' '):
    newimg = imgs[0]
    for i in imgs[1:]:
        newimg = np.hstack((newimg, i))
    
    cv2.imshow(title, newimg)
    cv2.waitKey(0)

import dbload

def show_distance():
    train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=20)
    num1 = train[labels==1]
    
    for i in range(1, len(num1)):
        print("distance between 0-{} {}".format(i, vector_dist(num1[0], num1[i])))
    
    for i in range(1, len(train)):
        print("distance between 0-{} {}".format(labels[i], vector_dist(num1[0], train[i])))
