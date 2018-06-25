import cv2
import os
import numpy as np
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
from keras.models import model_from_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('/home/ishita/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        face_cascade = cv2.CascadeClassifier('/home/ishita/opencv/data/lbpcascades/lbpcascade_frontalface.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        if (len(faces) == 0):
            return None, None
        x, y, w, h = faces[0]
        return gray[y:y+w, x:x+h], faces[0]
#return only the face part of the image
    x, y, w, h = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
 dirs = os.listdir(data_folder_path)
 #print(dirs)
 #list to hold all subject faces
 faces = []
 #list to hold labels for all subjects
 labels = []
 count=0
 #let's go through each directory and read images within it
 for dir_name in dirs:

    label = int(dir_name.replace("", ""))
    subject_dir_path = data_folder_path + "/" + dir_name
    subject_images_names = os.listdir(subject_dir_path)

    #for each subfolder
    for image_name in subject_images_names:
        image_path = subject_dir_path + "/" + image_name

        #read image
        image = cv2.imread(image_path)
        count+=1

        #display an image window to show the image
        #cv2.imshow("Training on image...", image)
        #cv2.waitKey(100)

        #detect face
        face, rect = detect_face(image)
        #------STEP-4--------
        #for the purpose of this tutorial
        #we will ignore faces that are not detected
        if face is not None:
            #add face to list of faces
            face=cv2.resize(face,(150,150))
            #print(face.shape," ",label)
            faces.append(face)
            #add label for this face
            labels.append(label)
 #cv2.waitKey(1)
 cv2.destroyAllWindows()
 print("no of counts:")
 print(count)
 return faces, labels


img_data, labels = prepare_training_data("training")
img_data = np.array(img_data)
img_data = img_data.astype('float32')
labels = np.array(labels ,dtype='int64')
img_data /= 255.0
img_data= np.expand_dims(img_data, axis=4)

#print(labels)
num_classes=3
encode = np_utils.to_categorical(labels,num_classes)
#print(Y)
x,y = shuffle(img_data,encode, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=20)
#print(x_test,y_test)

inputshape=img_data[0].shape
#print(input_shape)

#############################################################################
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=inputshape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

epochs = 10
lrate = 0.001
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=5,shuffle=True,verbose=2)
from keras.models import load_model
model.save('model.h5')
'''model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
#print("Loaded model from disk")'''
#
