import dlib
import cv2
import os
import glob
import imutils
import cv2
import os
import numpy as np
from skimage import io
from imutils import face_utils
from AlignDlib import AlignDlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
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

def face(image):
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    num_faces = len(rects)
    if num_faces == 0:
        return None
    return np.matrix([[p.x, p.y] for p in sp(gray, rects[0]).parts()])

def convex(image):
    x,y,z=image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    points=face(image)
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

image = cv2.imread('rach6.png')
convex(image)
