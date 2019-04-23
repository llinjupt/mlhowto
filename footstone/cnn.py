# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:10:34 2018

@author: Red
"""

import numpy as np
import dbload
import scaler
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

def test_cnn():
    '''Trains a simple convnet on the MNIST dataset.
    Gets to 99.25% test accuracy after 12 epochs
    (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
    '''
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras import backend as K
    
    batch_size = 128
    num_classes = 10
    epochs = 12
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    import crossvalid
    x_train, x_valid, y_train, y_valid = crossvalid.data_split(x_train, y_train, ratio=0.1, 
                                                               random_state=1)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=tf.nn.softmax))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])    

'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        320
_________________________________________________________________
activation_1 (Activation)    (None, 28, 28, 32)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                250890
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
=================================================================
Total params: 251,210
Trainable params: 251,210
Non-trainable params: 0
_________________________________________________________________
(3, 3, 1, 32)
(32,)
(25088, 10)
(10,)
'''
def cnn_model_create(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last"
    model = Sequential()
    input_shape = (height, width, depth)
    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)

    # define the first (and only) CONV => RELU layer
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model

'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               1179776
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
(3, 3, 1, 32)
(32,)
(3, 3, 32, 64)
(64,)
(9216, 128)
(128,)
(128, 10)
(10,)
'''
def cnn_model_create2(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation=tf.nn.softmax))
    
    return model

def cnn_load_mnist(ratio=1):
    num_classes = 10
    img_rows, img_cols = 28,28  
    x_train, y_train = dbload.load_mnist(r"./db/mnist", kind='train', 
                                   count=int(ratio*40000))
    x_test, y_test = dbload.load_mnist(r"./db/mnist", kind='test', 
                                 count=int(ratio*10000))

    x_train1, y_train1 = dbload.load_expand_mnist(offset_pixels=1)
    x_train2, y_train2 =  dbload.load_expand_rotate_mnist(degree=10)
    
    x_train = np.vstack([x_train, x_train1, x_train2])
    y_train = np.hstack([y_train, y_train1, y_train2])
    x_train, y_train = scaler.shuffle(x_train, y_train)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def cnn_mnist_plot(H, epochs):
    # plot the training loss and accuracy
    #plt.style.use("ggplot")
    plt.figure()

    plt.subplot(2,1,1)
    plt.title("Training Loss and Accuracy")
    plt.plot(np.arange(0, epochs), H.history["acc"], c='black', label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], c='gray', label="Validation")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')

    plt.subplot(2,1,2)
    plt.plot(np.arange(0, epochs), H.history["loss"], c='black', label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], c='gray', label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def cnn_weight_plot(ws):
    import drawutils,scaler
    
    bigmap = ws[2][0:25000,:].reshape(500,500)
    #for i in ws[1:]:
    #    bigmap = np.hstack([bigmap, i.flatten()])
    print(np.sum(bigmap[bigmap>=0]))
    print(np.sum(bigmap[bigmap<0]))
    bigmap = scaler.standard(bigmap)
    '''
    len = bigmap.shape[0]
    size = int(len ** 0.5)
    print(bigmap.shape, len, size)
    bigmap = bigmap[0:size**2].reshape(size,size)'''
    print(bigmap.shape)
    import cv2
    cv2.imshow("w", bigmap)
    cv2.waitKey()
    
    #drawutils.plot_showimgs([bigmap], 'Weight BigMap')
    
def cnn_mnist_test():
    import os
    from keras.callbacks import EarlyStopping
    from keras.models import load_model
    
    model_file = r'./models/mnist_expand.h51'
    x_train, y_train, x_test, y_test = cnn_load_mnist(1)
    
    # initialize the optimizer and model    
    if not os.path.exists(model_file):
        print("[INFO] compiling model...")
        
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
    
        # convert class vectors to binary class matrices
        print("[INFO] training network...")
        epochs = 5
        
        # SGD(lr=0.001)
        # keras.optimizers.Adadelta()
        model = cnn_model_create(width=28, height=28, depth=1, classes=10)
        model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(), 
                      metrics=["accuracy"])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        H = model.fit(x_train, y_train, validation_split=0.1, 
                      batch_size=256, epochs=epochs, verbose=1,
                      shuffle=True, callbacks=[early_stopping])       
        
        model.save(model_file)
        cnn_mnist_plot(H, epochs)
    else:
        print("[INFO] loading model...")
        model = load_model(model_file)

    ws = model.get_weights()
    cnn_weight_plot(ws)

    score = model.evaluate(x_test, y_test, verbose=1)       
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    cnn_mnist_test()
