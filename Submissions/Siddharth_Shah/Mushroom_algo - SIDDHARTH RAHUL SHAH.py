# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:41:54 2020

@author: sidsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv("train.csv");

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in Dataset.columns:
    Dataset[column] = le.fit_transform(Dataset[column])

X = Dataset.iloc[:,1:].values
Y = Dataset.iloc[:,0].values


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.15, random_state=1)

#importing libraries needed
from keras.models import Sequential 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dense 

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#classifier = build_classifier()
#classifier.fit(X_train,Y_train,epochs = 100,verbose =0,batch_size = 10)


classifier = KerasClassifier(build_fn = build_classifier, epochs = 100,batch_size=10)
classifier.fit(X_train,Y_train)


predictions = classifier.predict_proba(X_test)
Y_predicted = list()
for i in predictions:
    if i[0] > i[1]:
        Y_predicted.append(0)
    else:
        Y_predicted.append(1)
Y_predicted = np.asarray(Y_predicted)

count = 0;
for i in range(Y_predicted.shape[0]):
    if(Y_predicted[i] != Y_test[i]):
        count = count +1
print(count)
classifier.model.save("mushroom.model")

testset = pd.read_csv("Q1_Mushroom_test.csv");
let = LabelEncoder()
for column in testset.columns:
    testset[column] = let.fit_transform(testset[column])

predictions = classifier.predict_proba(testset)
Y_predicted = list()
for i in predictions:
    if i[0] > i[1]:
        Y_predicted.append('e')
    else:
        Y_predicted.append('p')
Y_predicted = np.asarray(Y_predicted)
with open('Q1.txt','w') as f:
    for i in Y_predicted:
        f.write("%s\n" % i)


 