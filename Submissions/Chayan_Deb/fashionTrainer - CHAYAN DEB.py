import pandas as pd
import numpy as np
import tensorflow as tf

trainData = pd.read_csv('train.csv')

y_train = np.array(trainData['label'], dtype=np.int)
x_train = np.array(trainData, dtype=np.float)[:,1:]

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.96):
            self.model.stop_training = True

callbacks = myCallBack()

x_train = x_train.reshape(x_train.shape[0],28,28,1)/255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28, 28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Input((x_train.shape[1])),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, callbacks=[callbacks])

testData = pd.read_csv('test.csv')
testData = np.array(testData, dtype=np.float)
testData = testData.reshape(testData.shape[0],28,28,1)/255.0
predictions = model.predict(testData)

with open('predictions.txt', 'w+') as f:
    for prediction in predictions:
        f.write(str(np.argmax(prediction)) + '\n')