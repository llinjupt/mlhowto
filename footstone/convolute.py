# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:32:45 2018

@author: Red
"""

import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

from numba import jit

@jit
def test_numba(size=10000):
    total = 0.0
    bigmatrix = np.ones((size,size))
    
    start = time.time()
    for i in range(bigmatrix.shape[0]):
        for j in range(bigmatrix.shape[1]):
            total += bigmatrix[i, j]
    print("bigmatrix sum cost walltime {:.02f}s".format(time.time()-start))
    return total

g_args = None
def arg_get(name):
    global g_args
    
    if g_args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=False, 
                        default=r"imgs/face.jpg",
                        help="path to input image")

        g_args = vars(ap.parse_args())
    
    return g_args[name]

'''
A = np.arange(16).reshape(4,4)
# 4%2==0, 4%4==0 must be satisfied : compatible with the shape
print(block_view(A, block=(2,4))) 
'''
# return a view with block on A (2D array)
def block_view(A, block=(3, 3)):
    from numpy.lib.stride_tricks import as_strided as ast
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
    strides= (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

# sum splitted blocks of a 2D array
def block_sum(A, block=(3,3)):
    if A.shape[0] % block[0] or A.shape[1] % block[1]:
        print("Not compatible shape")
        return None
    
    return block_view(A, block).sum(axis=(2,3))

# range is quicker more than twice np.arange speed
def test_range_speed(type=0, size=10000):
    import time
    
    start = time.time()
    if type == 0:
        for i in range(0, size):
            for j in range(0, size):
                pass
    else:
        for i in np.arange(0, size):
            for j in np.arange(0, size):
                pass
    print("cost walltime {} s".format(time.time() - start))

def show_matrix(matrix, color='black'):
    w, h = matrix.shape
    plt.figure(figsize=(w / 2 + 1, h / 2 + 1))

    index = 1
    for y in range(0,w):
        for x in range(0,h):
            ax = plt.subplot(w, h, index)
            
            # don't show x,y labels and axes
            plt.xlim(-1, 1) 
            plt.ylim(-1, 1)
            plt.xticks([])
            plt.yticks([])

            plt.text(0, 0, str(matrix[y,x]), fontsize=10,
                     verticalalignment="center",
                     horizontalalignment="center",
                     color=color)

            for i in ['top','bottom','left','right']:
                ax.spines[i].set_color(color) 

            index += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def chessboard(square=10, size=15, color=(255, 0, 0)):
    '''Create a chessboard color with order RGB'''
    
    # swap RGB to BGR for opencv
    color = color[::-1]
    
    base = np.zeros((square, square, 3), dtype='uint8')
    block0 = np.hstack(((base, (base + 1) * color))).astype(np.uint8)
    block1 = block0[:, ::-1, :]
    canvas = np.vstack((block0, block1))

    return np.tile(canvas, (size, size, 1))

def plt_showimgs(imgs, title=(),tight=True):
    plt.figure(figsize=(8,8))
    plt.title(title)
    
    count = len(imgs)
    columns = rows = int(count ** 0.5)
    if columns ** 2 < count:
        columns += 1
    
    if columns * rows < count:
        rows += 1
    
    index = 1
    for i in imgs:
        plt.subplot(rows, columns, index)
        plt.xticks([])
        plt.yticks([])
        
        if len(title) >= index:
            plt.title(title[index - 1]) 
        plt.imshow(i, cmap='gray', interpolation='none', vmin = 0, vmax = 255) 
        plt.axis('off')
        index += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    if tight:
        plt.tight_layout()
    plt.show()

def convolution_ignore_border(img, kernel):
    from skimage.exposure import rescale_intensity
    
    yksize, xksize = kernel.shape
    
    # kernel must with odd size
    if yksize % 2 == 0 or xksize % 2 == 0:
        print("kernel must with odd size")
        return None

    y_slide_count = img.shape[0] - kernel.shape[0]
    x_slide_count = img.shape[1] - kernel.shape[1]    
    if x_slide_count < 0 or y_slide_count < 0:
        print("img size too small to do convolution")
        return None

    newimg = img.copy().astype(np.float64)    

    # sliding kernel along y(right) and x(down) from left-top corner
    centery, centerx = yksize >> 1, xksize >> 1
    for y in range(0,y_slide_count+1):
        for x in range(0,x_slide_count+1):
            sum = (img[y:y+yksize,x:x+xksize] * kernel).sum()
            # round reducing truncation error float64 -> uint8
            newimg[y+centery, x+centerx] = round(sum)
    
    # rescale the output image in range [0, 255]
    newimg = rescale_intensity(newimg, in_range=(0, 255))
    return (newimg * 255).astype(np.uint8)

#@jit
def convolution(img, kernel):
    yksize, xksize = kernel.shape
    # kernel must with odd size
    if yksize % 2 == 0 or xksize % 2 == 0:
        print("kernel must with odd size")
        return None

    newimg = img.copy().astype(np.float64)
    y_slide_count,x_slide_count = img.shape

    left_right = (xksize - 1) // 2
    top_bottom = (yksize - 1) // 2
    img = cv2.copyMakeBorder(img, top_bottom, top_bottom, 
                             left_right, left_right, cv2.BORDER_REFLECT_101)

    # sliding kernel along y(right) and x(down) from left-top corner
    for y in range(0,y_slide_count):
        for x in range(0,x_slide_count):
            sum = (img[y:y+yksize,x:x+xksize] * kernel).sum()
            # round reducing truncation error float64 -> uint8
            newimg[y, x] = round(sum)
    
    # rescale the output image in range [0, 255]
    newimg = rescale_intensity(newimg, in_range=(0, 255))
    return (newimg * 255).astype(np.uint8)

def verify_convolution(size=8):
    np.random.seed(0)
    matrix = np.random.randint(0, 256, size=(size, size), dtype=np.uint8)
    kernel = np.ones((3, 3)) * 1.0 / 9
    newimg = convolution(matrix, kernel)
    print(np.all(newimg == cv2.filter2D(matrix, -1, kernel)))

def verify_kernels():
    smallblur = np.ones((7, 7), dtype=np.float64) * (1.0 / (7 * 7))
    largeblur = np.ones((21, 21), dtype=np.float64) * (1.0 / (15 * 15))  
    
    sharpen = np.array(([0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]), dtype=np.int32)
    
    laplacian = np.array(([0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]), dtype=np.int32)
    sobelX = np.array(([-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]), dtype=np.int32)
    sobelY = np.array(([-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]), dtype=np.int32)
    
    emboss = np.array(([-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]), dtype=np.int32)
    
    kernels = (smallblur, largeblur, sharpen, laplacian, sobelX, sobelY, emboss)
    labels = ('smallBlur', 'largeBlur', 'Sharpen', 'Laplacian', 'sobelX', 'sobelY', 'Emboss')
    
    image = cv2.imread(arg_get("image"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    imgs = []
    for kernel in kernels:
        newimg = convolution(gray, kernel)
        imgs.append(newimg)
        
    plt_showimgs(imgs, labels, tight=False)

def convolution_fast(img, kernel):
    yksize, xksize = kernel.shape
    # kernel must with odd size
    if yksize % 2 == 0 or xksize % 2 == 0:
        print("kernel must with odd size")
        return None

    newimg = img.copy().astype(np.float64) * 0

    # extend four borders to convolute border pixels
    left_right = (xksize - 1) // 2
    top_bottom = (yksize - 1) // 2
    img = cv2.copyMakeBorder(img, top_bottom, top_bottom, 
                             left_right, left_right, cv2.BORDER_REFLECT_101)
  
    # extend kernel as possible as the img size, but no bigger than img
    ytile = img.shape[0] // yksize
    xtile = img.shape[1] // xksize
    nkernel = np.tile(kernel, (ytile, xtile))
    
    # sliding kernel along y(right) and x(down) from left-top corner
    ynksize, xnksize = nkernel.shape
    for y in range(0, yksize):
        for x in range(0, xksize):
            # use nkernel convolute img, so have a cross window 
            w_window = min([img.shape[0] - y, ynksize])
            h_window = min([img.shape[1] - x, xnksize])
            
            # resize the window round base kernel size
            (ny, ry) = divmod(w_window, yksize)
            (nx, rx) = divmod(h_window, xksize)
            
            w_window  -= ry
            h_window  -= rx
            
            tmp = img[y:w_window+y, x:h_window+x] * nkernel[:w_window, :h_window]
            tmp = tmp.reshape(ny, yksize, nx, xksize).sum(axis=(1, 3))

            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    newimg[y + i * yksize, x + j * xksize] = round(tmp[i,j])

    # rescale the output image in range [0, 255]
    newimg = rescale_intensity(newimg, in_range=(0, 255))
    return (newimg * 255).astype(np.uint8)

def convolute_speed_cmp(image=None, count=100, type=0):
    if image is None:
        image = cv2.imread(arg_get("image"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    kernel = np.ones((3, 3)) * 1.0 / 9
    start = time.time()
    if type == 0:    
        for i in range(0,count):
            convolution_fast(gray + count, kernel)
        print("convolution_fast cost walltime {:.02f}s with loop {}".format(time.time()-start, count))
    elif type == 1:
        for i in range(0,count):
            cv2.filter2D(gray + count, -1, kernel)
        print("filter2D cost walltime {:.02f}s with loop {}".format(time.time()-start, count))
    else:
        for i in range(0,count):
            convolution(gray + count, kernel)
        print("convolution cost walltime {:.02f}s with loop {}".format(time.time()-start, count))        

convolute_speed_cmp(None, 10, 2)
