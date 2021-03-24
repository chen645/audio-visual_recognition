#!/usr/bin/env python
# coding: utf-8

# In[15]:




# In[16]:


import librosa
from librosa import display
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import time
import numpy as np


# In[7]:


def extract_mfcc(file):
    #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
  # If the file is not valid, skip it
    
    return mfccs


# In[86]:


class_map = {'angry':0, 'disgust':1, 'fearful':2,'happy':3,'neutral':4,'sad':5,'surprised':6}
def data_extract(data):
    data_path = data+'/*/*.wav' 
    img_paths = glob(data_path)
    data_count = len(glob(data_path))
    print(data_count)

    X = np.zeros((data_count, 40))
    y = np.zeros((data_count, ))
    for i, path in tqdm(sorted(enumerate(img_paths))):

        X[i] = extract_mfcc(path)

        cls = path.split('/')[-2]

        y[i] = class_map[cls]
    return X,y


# # Train data Extract

# In[118]:

import joblib
import keras
'''
X,y =data_extract('train') 


# In[119]:



X_name = 'X_train.joblib'
y_name = 'y_train.joblib'

y_onehot = keras.utils.to_categorical(y, num_classes=7)
print(y.shape)
savedX = joblib.dump(X,X_name)
savedy = joblib.dump(y_onehot, y_name)
'''

# In[147]:


X_train = joblib.load('X_train.joblib')
y_train = joblib.load('y_train.joblib')



x_traincnn = np.expand_dims(X_train, axis=2)
print(x_traincnn.shape,y_train.shape)
print(y_train[0])
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.python.framework import tensor_util

from tensorflow.keras.layers import Conv1D, MaxPooling1D,LSTM,LeakyReLU,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv1D(16, 5,padding='same',
                 input_shape=(40,1)))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(32, 5,padding='same'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(64, 5,padding='same'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(128,5,padding='same'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(256, 5,padding='same'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())

model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()


# In[149]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(x_traincnn.shape, y_train.shape)
'''
model.fit(x_traincnn, y_train, batch_size=128, epochs=1000)

model.save('ASR1.h5') 
# # Test data Extract
'''
# In[91]:





# In[92]:

model.load_weights('ASR1.h5')
import joblib
import keras

X_name = 'X_test.joblib'
y_name = 'y_test.joblib'
'''
x_testcnn,y_test =data_extract('test') 
y_onehot = keras.utils.to_categorical(y_test, num_classes=7)
joblib.dump(x_testcnn,X_name)
joblib.dump(y_onehot, y_name)
'''

# In[95]:


x_testcnn = joblib.load('X_test.joblib')
y_test = joblib.load('y_test.joblib')
prediction_name = 'speech_prediction'
x_testcnn = np.expand_dims(x_testcnn, axis=2)
from sklearn.metrics import classification_report,confusion_matrix
y_pred = np.argmax(model.predict(x_testcnn), axis=-1)
y_test = np.argmax(y_test, axis=-1)
savedprediction = joblib.dump(y_pred, prediction_name)
target_names = ['angry', 'disgust', 'fearful','happy','neautral','sad','surprised']
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test,y_pred))


# In[ ]:




