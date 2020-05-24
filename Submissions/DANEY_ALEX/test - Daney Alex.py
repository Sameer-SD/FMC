import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
# import the necessary packages
from pyimagesearch.minivggnet import MiniVGGNet
from pyimagesearch.fashionModel import FashionModel

from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras import utils as  np_utils
from tensorflow.keras import backend as K
from imutils import build_montages
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import pandas as pd

filename1=sys.argv[1]
images = []
test=pd.read_csv(filename1)
testX=np.asarray(test)
model=keras.models.load_model('FashionModel.h5')

if K.image_data_format() == "channels_first":
    testX = reshape((testX.shape[0], 1, 28, 28))

else:
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
 
# scale data to the range of [0, 1]
testX = testX.astype("float32") / 255.0


testY=[]
labelNames = ["top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot"]

for j in range(0,len(testX)):
    # classify the clothing
    probs = model.predict(testX)
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[j]]
    testY.append(str(label))
    print(testY)
print(testY)

with open('Question2.txt', 'w') as f1:
    for i in range (0,len(testX)):
        line=testY[i]+"\n"
        f1.write(line)
        print(i)
