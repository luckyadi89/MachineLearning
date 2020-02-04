#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[2]:


data = load_breast_cancer()

X_train,X_test, y_train,y_test = train_test_split(data.data, data.target, test_size=0.2)

N,D = X_train.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = Sequential()
model.add(Input(shape = (D,)))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

r = model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs = 100)

print("Train_score:  ",model.evaluate(X_train,y_train,verbose = 0))
print("Test_score: ", model.evaluate(X_test,y_test,verbose = 0))

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val-loss')
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()

y_pred = model.predict_classes(X_test)
y_pred
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
