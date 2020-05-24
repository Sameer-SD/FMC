import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.999):
            self.model.stop_training = True
callbacks = myCallBack()

trainData = pd.read_csv('train.csv')
le = LabelEncoder()
trainData['class'] = 1 - le.fit_transform(trainData['class'])   # Now 1 represents edible mushrooms and 0 represents non-edible mushrooms

testData = pd.read_csv('test.csv')
df = pd.concat((trainData,testData), sort=False)

df = pd.get_dummies(df)

y_train = np.array(df['class'], dtype=np.float)[0:trainData.shape[0]]
x_train = np.array(df, dtype=np.float)[0:trainData.shape[0],1:]

print(y_train.shape, x_train.shape)

input_layer = tf.keras.layers.Input((x_train.shape[1]))
hidden_layer = tf.keras.layers.Dense(64, activation='relu')(input_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


testing = np.array(df, dtype=np.float)[trainData.shape[0]:,1:]
predictions = model.predict(testing)
predictions = [int(prediction>0.8) for prediction in predictions]

with open('prediction.txt', 'w+') as f:
    for label in predictions:
        if label == 1:
            f.write('e\n')
        else :
            f.write('p\n')