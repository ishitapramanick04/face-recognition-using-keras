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
def prepare_training_data(data_folder_path):
 dirs = os.listdir(data_folder_path)
 faces = []
 labels = []

 for dir_name in dirs:
    label = int(dir_name.replace("", ""))
    subject_dir_path = data_folder_path + "/" + dir_name
    subject_images_names = os.listdir(subject_dir_path)
    count=0
    x=0
    #for each subfolder
    for image_name in subject_images_names:
        image_path = subject_dir_path + "/" + image_name
        image = cv2.imread(image_path)
        count+=1
        face1 = convex(image)
        if face1 is not None:
            #add face to list of faces

            face=detect_face(face1)
            if face is not None:
                face=cv2.resize(face,(180,180))
                x+=1
                faces.append(face)
                labels.append(label)
                cv2.imshow(str(label),face)
                cv2.waitKey(1)
    print("count: ",count," detected: ",x)
 return faces, labels

img_data, labels = prepare_training_data("training")
img_data = np.array(img_data)
img_data = img_data.astype('float32')
labels = np.array(labels ,dtype='int64')
img_data /= 255.0
#img_data= np.expand_dims(img_data, axis=4)

#print(labels)
num_classes=3
encode = np_utils.to_categorical(labels,num_classes)
#print(encode)
x,y = shuffle(img_data,encode, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
#print(x_test,y_test)

inputshape=img_data[0].shape
print(inputshape)

#############################################################################
from keras.applications import VGG16
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=inputshape)
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
model = Sequential()
model.add(vgg_conv)

# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

epochs = 10
lrate = 0.001
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=10,shuffle=True,verbose=2)
from keras.models import load_model
model.save('modeldlib.h5')
