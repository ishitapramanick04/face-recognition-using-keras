import dlib
import cv2
import os
import glob
from skimage import io
import imutils
from imutils import face_utils
import cv2
import os
import numpy as np
from imutils.face_utils.facealigner import FaceAligner
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import sys
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
from keras.models import load_model
from AlignDlib import AlignDlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def faces(img):
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa=FaceAligner(sp)
    #img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 2)
    num_faces = len(rects)
    if num_faces == 0:
        return None
    b=0
    l=0
    for rect in rects:
    	# extract the ROI of the *original* face, then align the face
    	# using facial landmarks
    	(x, y, w, h) = face_utils.rect_to_bb(rect)
    	#print(w," ",h)
    	if w > b and h > l:
    		b=w
    		l=h
    		faceAligned = rect
    return np.matrix([[p.x, p.y] for p in sp(gray, faceAligned).parts()])
def convex(image):
    x,y,z=image.shape
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    points=faces(image)
    if points is not None:
        jaw=points[0:17]
        jaw_rev=np.flip(jaw,0)
        right=points[23:27]
        left=points[18:26]
        arr=np.concatenate((left,right,jaw_rev))
        #arr=points[0:27]
        #print(arr.shape)
        #print(jaw,jaw_rev)
        img = np.zeros((x,y,z), np.uint8)
        cv2.polylines(img,[arr],True,(255,255,255))
        cv2.fillPoly(img, pts =[arr], color=(255,255,255))
        img = cv2.bitwise_and(image,img,img)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        return img
    return None
def detect_face(img):
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa=FaceAligner(sp)
    #img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 2)
    num_faces = len(rects)
    if num_faces == 0:
        return None
    faceAligned = fa.align(img, gray, rects[0])
    #image = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
    return faceAligned

def test_data(data_folder_path):
 dirs = os.listdir(data_folder_path)
 new=load_model('modeldlib.h5')
 for dir_name in dirs:
        test_dir_path = data_folder_path + "/" + dir_name
        image = cv2.imread(test_dir_path)
        face1 = convex(image)
        if face1 is not None:
            face=detect_face(face1)
            if face is not None:
                f=face
                face=cv2.resize(face,(180,180))
                face=np.array(face)
                face = face.astype('float32')
                face /= 255.0
                #print(face.shape)
                face= np.expand_dims(face, axis=0)
                #print(face.shape)
                pred =  new.predict_classes(face)
                print(test_dir_path,"  ",pred)
                cv2.imshow(str(pred),f)
                cv2.waitKey(1)
#cv2.destroyAllWindows()
test_data('test')
