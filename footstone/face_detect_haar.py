# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:32:45 2019

@author: Red
"""

import numpy as np
import cv2
import os

def img_resize(img, ratio=0.5, inter=cv2.INTER_AREA):
    w = img.shape[1] * ratio
    h = img.shape[0] * ratio
    
    return cv2.resize(img, (int(w), int(h)), interpolation=inter)

def fimg_resize(sfile, dfile='resize', ratio=0.5, inter=cv2.INTER_AREA):
    img = cv2.imread(sfile)
    newimg = img_resize(img, ratio, inter=inter)
    
    # generate dst file name
    abspath = os.path.abspath(sfile)
    sfile = os.path.basename(abspath)
    dirname = os.path.dirname(abspath)
    
    ext = os.path.splitext(sfile)[-1]
    dfile = os.path.join(dirname, dfile + ext)
    
    cv2.imwrite(dfile, newimg)

def isgray(img):
    if len(img.shape) == 2:
        return True
    return False

def rotate(image, angle):
    '''roate image around center of image'''
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

#fimg_batch_resize("./imgs/face.jpg")
def fimg_batch_resize(sfile, count=10):
    for i in range(1, count + 1,1):
        ratio = i * 0.1
        dfile = ("%02d" % i)
        fimg_resize(sfile, dfile, ratio=ratio)

# load opencv to handle image
import cv2
import argparse

def args_handle():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, 
                    default=r"imgs/face.jpg",
                    help="path to input image")

    ap.add_argument("-m", "--model", required=False,
                    default=r"models/haarcascades/haarcascade_frontalface_default.xml",
                    help="path to opencv haar pre-trained model")
    
    return vars(ap.parse_args())

g_args = None
def arg_get(name):
    global g_args
    
    if g_args is None:
        g_args = args_handle()
    return g_args[name]

class FaceDetector():
    def __init__(self, model):
        self.faceClassifier = cv2.CascadeClassifier(model)
    
    def detect_img(self, img):
        gray = img if isgray(img) else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.faceClassifier.detectMultiScale(gray, 
                                                    scaleFactor=1.5, 
                                                    minNeighbors=5, 
                                                    minSize=(30,30))
    def detect_fimg(self, fimg, verbose=0):
        # load jpg file from disk
        image = cv2.imread(fimg)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faceRects = self.detect_img(gray)

        # draw rects on image and show up
        for x,y,w,h in faceRects:
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2) 
        
        return image
    
    def show_and_wait(self, image, title=' '):
        cv2.imshow(title, image)
        cv2.waitKey(0)

def face_batchdetect_haar_size(model_path, fimg):
    import time
    img = cv2.imread(fimg)
    FD = FaceDetector(model_path)
    for i in range(1, 21, 1):
        ratio = i * 0.1
        newimg = img_resize(img, ratio, inter=cv2.INTER_AREA)

        # time cost
        start = time.process_time()
        for i in range(0, 10):
            faceRects = FD.detect_img(newimg)
        end = time.process_time()
        
        faces = len(faceRects)
        print("I found {} face(s) of ratio {:.2f} with shape{} cost time {:.2f}".format(faces, 
              ratio, newimg.shape, end - start))

        for x,y,w,h in faceRects:
            cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
        if faces != 2 and faces != 0:
            FD.show_and_wait(newimg, "{:.2f}".format(ratio))

def face_detect_camera(model_path, show=0):
    import time
    frames = 0
    camera = cv2.VideoCapture(0)
    start = time.process_time()
    
    FD = FaceDetector(model_path)
    while(True):
        grabbed, frame = camera.read()
        
        if not grabbed:
            print("grabbed nothing, just quit!")
            break

        faceRects = FD.detect_img(frame)
        frames += 1
        
        fps = frames / (time.process_time() - start)
        print("{:.2f} FPS".format(fps), flush=True)
        
        if not show:
            continue
        
        cv2.putText(frame, "{:.2f} FPS".format(fps),
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for x,y,w,h in faceRects:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 2)
    
        cv2.imshow("Face", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

def face_batchdetect_haar_rotate(model_path, fimg):
    import time
    img = cv2.imread(fimg)
    FD = FaceDetector(model_path)
    for angle in range(0, 11, 1):
        newimg = rotate(img, angle)
        # time cost
        start = time.process_time()
        for i in range(0, 10):
            faceRects = FD.detect_img(newimg)
        end = time.process_time()
        
        faces = len(faceRects)
        print("I found {} face(s) of rotate {} with shape{} cost time {:.2f}".format(faces, 
              angle, newimg.shape, end - start))
        
        for x,y,w,h in faceRects:
            cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
        if faces != 2 and faces != 0:
            FD.show_and_wait(newimg, "Rotate{}".format(angle))

def translation(image, x, y):
    '''move image at x-axis x pixels and y-axis y pixels'''
    
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def face_batchdetect_haar_move(model_path, fimg):
    import time
    img = cv2.imread(fimg)
    FD = FaceDetector(model_path)
    for move in range(0, 100, 10):
        newimg = translation(img, move, move)
        
        # time cost
        start = time.process_time()
        for i in range(0, 10):
            faceRects = FD.detect_img(newimg)
        end = time.process_time()

        faces = len(faceRects)
        print("I found {} face(s) of move {} with shape{} cost time {:.2f}".format(faces, 
              move, newimg.shape, end - start))
        
        for x,y,w,h in faceRects:
            cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
        #if faces != 1 and faces != 0:
        FD.show_and_wait(newimg, "Move{}".format(move))

#model_path = arg_get('model')
#face_batchdetect_haar_size(model_path, arg_get('image'))
#face_batchdetect_haar_rotate(model_path, arg_get('image'))
#face_batchdetect_haar_move(model_path, arg_get('image'))

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
