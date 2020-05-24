# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:01:00 2020

@author: sidsh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv("fashion-mnist_train.csv");
X = Dataset.iloc[:,1:].values
Y = Dataset.iloc[:,0].values

X_img = list()

for i in range(X.shape[0]):
    X_img.append((X[i][:]).reshape(28,28))
    
from keras.utils import np_utils
Y = np_utils.to_categorical(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_img,Y,test_size = 0.167, random_state=0)

X_train = (np.asarray(X_train).astype('float32'))
X_train = X_train.reshape(X_train.shape[0],28,28,1)
Y_train = np.asarray(Y_train)
X_test = (np.asarray(X_test).astype('float32'))
X_test = X_test.reshape(X_test.shape[0],28,28,1)
Y_test = np.asarray(Y_test)


X_train = X_train/255
X_test = X_test/255
plt.imshow(X_train[0].reshape(28,28))

num_classes = Y_train.shape[1]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

def define_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15,(3,3),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(50,activation = 'relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = define_model()
model.fit(X_train, Y_train,epochs = 10,batch_size = 200,verbose = 2)

scores = model.evaluate(X_test,Y_test,verbose = 0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


testset = (pd.read_csv("Q2_Clothing_test.csv")).values;
X_img_test = list()

for i in range(testset.shape[0]):
    X_img_test.append((testset[i][:]).reshape(28,28))
x_test = (np.asarray(X_img_test)).reshape(testset.shape[0],28,28,1)
plt.imshow(x_test[0].reshape(28,28))

predictions = model.predict(x_test)
predictions = model.predict_classes(x_test)
with open("Q2.txt",'w') as f:
    for i in predictions:
        f.write("%s\n" % i)

#model.save("fashionmodel.model")
    