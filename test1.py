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
def detect_face(img):
    fa=AlignDlib('shape_predictor_68_face_landmarks.dat')
    facealign=fa.align(180,img)
    if facealign is not None:
        gray = cv2.cvtColor(facealign, cv2.COLOR_BGR2GRAY)
    else:
        return None
    return gray


def test_data(data_folder_path):
 dirs = os.listdir(data_folder_path)
 new=load_model('modeldlib.h5')
 for dir_name in dirs:
        test_dir_path = data_folder_path + "/" + dir_name
        image = cv2.imread(test_dir_path)
        face = detect_face(image)
        if face is not None:
            f=face
            face=cv2.resize(face,(180,180))
            face=np.array(face)
            face = face.astype('float32')
            face /= 255.0
            face= np.expand_dims([face], axis=4)
            pred =  new.predict_classes(face)
            print(test_dir_path,"  ",pred)
            cv2.imshow(str(pred),f)
            cv2.waitKey(1)
#cv2.destroyAllWindows()
test_data('test')
