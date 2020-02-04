#!/usr/bin/env python
# coding: utf-8

# In[1]:


#this time instead of importing things beforehand, we will import as we go along
import tensorflow as tf
import numpy as np


# In[2]:


mnist= tf.keras.datasets.mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.astype('float')
X_test = X_test.astype('float')
X_train, X_test = X_train/255.0 , X_test/255.0



print(X_train.shape, X_test.shape, y_train.shape,y_test.shape)


# In[3]:


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

r = model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs = 10)


# In[5]:


import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val-loss')
plt.legend()

