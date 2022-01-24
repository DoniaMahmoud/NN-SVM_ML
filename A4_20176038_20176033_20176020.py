from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import os

import cv2
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics

def normalize(data):
    x=[]
    for i in data:
        result=i/255.0
        x.append(result)
    x=np.array(data)
    return x

imagePaths = list(paths.list_images("Dataset"))

data = []
labels = []
for imagePath in imagePaths:
    image = load_img(imagePath)
    imgGray = image.convert('L')
    label = imagePath.split(os.path.sep)[-2]
    imgGray = img_to_array(imgGray)
    data.append(imgGray)
    labels.append(label)

labels = np.array(labels)
new_labels = labels.astype(np.int)
SVMlabels = new_labels
new_labels = new_labels.reshape(-1,1)
afterNormalize=normalize(data)
afterNormalize=afterNormalize.reshape(2059,10000)


ohe = OneHotEncoder()
y_encoded = ohe.fit_transform(new_labels).toarray()


X_train,X_test,y_train,y_test = train_test_split(afterNormalize,y_encoded,test_size = 0.20,shuffle=True)


X_train = np.array([X_train]).reshape(1647,10000)
Y_train = np.array([y_train]).reshape(1647,10)
X_test = np.array([X_test]).reshape(412,10000)
Y_test = np.array([y_test]).reshape(412,10)


#Neural network1
model1 = Sequential()
model1.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model1.add(Dense(256, activation='relu'))
model1.add(Dense(128, activation='relu'))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(y_train.shape[1], activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained= model1.fit(X_train, y_train, epochs=65)
accuracy1=model1.evaluate(X_test, y_test)



#Neural network2
model2 = Sequential()
model2.add(Dense(600, input_dim=X_train.shape[1], activation='relu'))
model2.add(Dense(300, activation='relu'))
model2.add(Dense(150, activation='relu'))
model2.add(Dense(y_train.shape[1], activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained2= model2.fit(X_train, y_train, epochs=60)
accuracy2=model2.evaluate(X_test, y_test)
print("Model 1 accuracy=",accuracy1[1]*100)
print("Model 2 accuracy=",accuracy2[1]*100)



#SVM
SVM_X_train, SVM_X_test, SVM_Y_train, SVM_Y_test = train_test_split(afterNormalize, SVMlabels, test_size=0.2,shuffle=True)
clf = svm.SVC(kernel='linear')
clf.fit(SVM_X_train, SVM_Y_train)
y_pred3 = clf.predict(SVM_X_test)
SVMAccuracy3 = metrics.accuracy_score(SVM_Y_test, y_pred3)
print("Model 3 accuracy=",SVMAccuracy3*100)


max=0
if(accuracy1[1]>accuracy2[1] and accuracy1[1]>SVMAccuracy3):
    max=accuracy1[1]
    print("Neural Network 1 is the best of 3 models with accuracy=", max*100,"%")
elif(accuracy2[1]>accuracy1[1] and accuracy2[1]>SVMAccuracy3):
    max = accuracy2[1]
    print("Neural Network 2 is the best of 3 models with accuracy=", max * 100, "%")
elif(SVMAccuracy3>accuracy1[1] and SVMAccuracy3>accuracy2[1]) :
    max = SVMAccuracy3
    print("SVM is the best of 3 models with accuracy=", max * 100, "%")