#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
'''
import matplotlib.pyplot as plt
'''
import numpy as np
from tqdm.auto import tqdm
from keras import backend as K
import keras
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model
from sklearn.model_selection import train_test_split

import tensorflow as tf
import gc

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# In[2]:


IMG_SIZE = 314
class_map = {'angry':0, 'disgust':1, 'fearful':2,'happy':3,'neautral':4,'sad':5,'surprised':6}


# In[3]:

'''
import os
# Read single image
img_paths = 'real_40images/*/*jpg'
img_paths = glob(img_paths) 
class_name=os.listdir('real_40images')
print(class_name)

 # demo for 200 images
img = cv2.imread(img_paths[0])
plt.imshow(img)


# In[ ]:
'''
'''
data_count = len(img_paths)
X = np.zeros((data_count, IMG_SIZE, IMG_SIZE, 3))
y = np.zeros((data_count, ))
'''

# In[ ]:
''''

for i, path in tqdm(enumerate(img_paths)):
    img = cv2.imread(path)
    img_resize = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cls = path.split('\\')[-2]
    print(path)
    X[i] = img_resize
    y[i] = class_map[cls]
'''

# In[ ]:
'''

X = X/255
y_onehot = keras.utils.to_categorical(y, num_classes=7)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=5566)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
'''

# In[3]:


# base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
# x = base_model.output
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(7, activation='softmax')(x)
# model = Model(base_model.input, predictions)


# In[4]:


# import os
# path='real_40images'
# weight = 'emotion_image.h5'
# model.load_weights(weight)
# img_paths = 'real_40images/*/*jpg'
# img_paths = glob(img_paths) 
# data_count = len(img_paths)
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# print(data_count)
# data=[]
# for i in os.walk(path):
    
#     data.append(i)
# print(data[0][1][0])
# for i in data[0][1]:
#     print(i)


# In[21]:






'''
def glob_image(image):
    x = glob('real_40images/'+image+'/*.jpg')
    data_count=len(x)
    return x,data_count


 # demo for 200 images
print(data_count)
for count in data[0][1]:
    x, data_count = glob_image(count)
    i=0
    X=np.zeros((int(data_count/10),IMG_SIZE,IMG_SIZE,3))
    y=np.zeros((int(data_count/10),))
    number=0
    for j, path in tqdm(enumerate(x)):

        img = cv2.imread(path)

        img_resize = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X[i]=img_resize/255
        
        i+=1
        
        print(i)
        if ((int(j+1)%(int(data_count/10))==0)&j!=0):

            X=np.argmax(model.predict(X), axis=-1)
            np.save(str(count)+"_train"+str(number),X)
            del X  #clear X memory
            i=0
            number+=1
            gc.collect() 
            X=np.zeros((int(data_count/10),IMG_SIZE,IMG_SIZE,3))

'''
    



# In[5]:


# y= np.zeros((data_count,))
# for i, path in tqdm(enumerate(img_paths)):
#     cls = path.split('\\')[-2]
#     y[i] = class_map[cls]
    


# In[8]:


# print(y[0])


# In[11]:


# X=[]
# for count in data[0][1]:
#     for i in range(10):
#         X1=[]
#         X1=np.load(str(count)+"_train"+str(i)+".npy")
#         X1 = np.array(X1)
#         print(len(X1))
#         for i in X1:
#             X.append(i)
            
        


# In[4]:


# from sklearn.metrics import classification_report, confusion_matrix


# In[18]:


# X_lstm = X.reshape(int(len(X)/40),40)
# print(X_lstm)

# y_lstm = np.zeros(int(len(y)/40),)
# i=0
# for j in range(len(y)):
#     if(j%40==0):
#         y_lstm[i]=int(y[j])
#         i+=1
# np.save("X_lstm_train",X_lstm)
# np.save("y_lstm_train",y_lstm)


# In[5]:


# X_lstm=np.load("X_lstm_train.npy")
# y_lstm=np.load("y_lstm_train.npy")
# y_lstm_onehot = keras.utils.to_categorical(y_lstm, num_classes=7)
# X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm_onehot, test_size=0.2, random_state=5566)
# # X_train.shape, X_test.shape, y_train.shape, y_test.shape
# print(X_train)


# In[6]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.layers.recurrent import SimpleRNN
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten,Dropout


# In[7]:



# model1 = Sequential()
# model1.add(Embedding(output_dim=128, input_dim=7, input_length=40))

# model1.add(Dropout(0.25))

# model1.add(Bidirectional(LSTM(32,return_sequences=True)))
# model1.add(Bidirectional(LSTM(32,return_sequences=True)))
# model1.add(Bidirectional(LSTM(32,return_sequences=False)))
# model1.add(Dense(7))
# model1.add(Activation('softmax'))
# opt = keras.optimizers.Adam(learning_rate=0.01)
# model1.compile(loss='mse', optimizer=opt,
#               metrics=['accuracy'])
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# checkpoint = ModelCheckpoint('video_40images.h5',
#     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)


# In[9]:


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,classification_report
import time 
fold_num=10
kfold = KFold(n_splits=fold_num,shuffle=True,random_state=5576)
results=0
predict_result=[]
X_lstm=np.load("X_lstm_train.npy")
y_lstm=np.load("y_lstm_train.npy")
y_lstm = keras.utils.to_categorical(y_lstm, num_classes=7)
a = time.time()
for train,test in kfold.split(X_lstm):
    x_train,x_test=X_lstm[train],X_lstm[test]
    y_train,y_test=y_lstm[train],y_lstm[test]


    model1 = Sequential()
    model1.add(Embedding(output_dim=128, input_dim=7, input_length=40))

    model1.add(Dropout(0.25))

    model1.add(Bidirectional(LSTM(32,return_sequences=True)))
    model1.add(Bidirectional(LSTM(32,return_sequences=True)))
    model1.add(Bidirectional(LSTM(32,return_sequences=False)))
    model1.add(Dense(7))
    model1.add(Activation('softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model1.compile(loss='mse', optimizer=opt,
              metrics=['accuracy'])
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    checkpoint = ModelCheckpoint('video_40images.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)



    model1.fit(x_train, y_train,
                batch_size=2048,
                epochs=100,
                callbacks=[reduce_lr],
                verbose=1,    
                validation_data=(x_test, y_test))
    print(time.time()-a)

   
print(time.time()-a)
print("平均測試率:(%.3f)"%(results/fold_num))
print(time.time()-a)  


# In[39]:


model1.save('image_lstm.h5')
from sklearn.metrics import classification_report, confusion_matrix
y_true = np.argmax(y_test, axis=-1)
y_pred = np.argmax(model1.predict(X_test), axis=-1)
print(y_true.shape, y_pred.shape)
target_names = [str(i) for i in range(7)]
print(classification_report(y_true, y_pred, target_names=target_names))
print(confusion_matrix(y_true, y_pred))


# In[6]:



weight_1 = 'image_lstm.h5'
model1.load_weights(weight_1)
img_paths = 'real_40images_test/*/*jpg'
img_paths = glob(img_paths) 
data_count = len(img_paths)


# In[42]:


def glob_image(image):
    x = glob('real_40images_test/'+image+'/*.jpg')
    data_count=len(x)
    return x,data_count


 # demo for 200 images
print(data_count)
for count in data[0][1]:
    x, data_count = glob_image(count)
    i=0
    X=np.zeros((int(data_count/10),IMG_SIZE,IMG_SIZE,3))
    y=np.zeros((int(data_count/10),))
    number=0
    for j, path in tqdm(enumerate(x)):

        img = cv2.imread(path)

        img_resize = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X[i]=img_resize/255
        
        i+=1
        del img_resize,path
        print(i)
        if ((int(j+1)%(int(data_count/10))==0)&j!=0):

            X=np.argmax(model.predict(X), axis=-1)
            np.save(str(count)+"_test"+str(number),X)
            del X
            i=0
            number+=1
            gc.collect()
            X=np.zeros((int(data_count/10),IMG_SIZE,IMG_SIZE,3))


# In[57]:


data_count = len(img_paths)
y= np.zeros((data_count,))
for i, path in tqdm(enumerate(img_paths)):
    cls = path.split('\\')[-2]
    y[i] = class_map[cls]
X=[]
for count in data[0][1]:
    for i in range(10):
        X1=[]
        X1=np.load(str(count)+"_test"+str(i)+".npy")
        X1 = np.array(X1)
    
        for i in X1:
            X.append(i)
            
X=np.array(X)
X.shape
            
        


# In[58]:



X_lstm_test = X.reshape(int(len(X)/40),40)
print(X_lstm_test)

y_lstm_test = np.zeros(int(len(y)/40),)
i=0
for j in range(len(y_lstm_test)):
    if(j%40==0):
        y_lstm_test[i]=int(y[j])
        
        i+=1
np.save("X_lstm_test",X_lstm_test)
np.save("y_lstm_test",y_lstm_test)


# In[10]:


X_lstm_test=np.load("X_lstm_test.npy")
y_lstm_test=np.load("y_lstm_test.npy")
y_pred = np.argmax(model1.predict(X_lstm_test), axis=-1)

target_names = ['angry', 'disgust', 'fearful','happy','neautral','sad','surprised']
print(classification_report(y_lstm_test, y_pred, target_names=target_names))
print(confusion_matrix(y_lstm_test, y_pred))


# In[ ]:




