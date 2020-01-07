#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import wget
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Dense
from tensorflow import math as tfmth
from tensorflow.keras.callbacks import LearningRateScheduler


os.remove('moore.csv')


url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/linear_regression_class/moore.csv'


filename = wget.download(url)

data = pd.read_csv(filename,sep='\t', header = None)
X = data[2]
y = data[1]

#Data Cleaning
X = list(map(lambda P: re.findall('\d{4}',P),X))
y = list(map(lambda P: ''.join(re.findall('[^\D]',P.split('[')[0])),y))

# converting to numpy array
X = np.array(X).astype(int).reshape(-1,1)
y = np.array(y).astype(float).reshape(-1)


plt.scatter(X,y)



# ### We realise that it is logarithmic of the type y = Ar^x
# ### we can simplify it to logy = logA + Xlogr
# ### looks like a linear function now y = mX + b
# 
# ### So we convert y into log y
# 

# In[2]:


y = np.log(y)

plt.scatter(X,y)


# ### Now this is LinearRegression as the data is  kind of linear now

# In[3]:


## let us center X as well so that values of X arent too large
X = X - np.mean(X)
print(X)


# In[4]:


# now we build a model

model = Sequential()

model.add(Input(shape=(1,)))
model.add(Dense(1,))

def scheduler(epoch):
    
    if epoch > 50:
        return 0.0001
    else:
        return 0.001 

callback = LearningRateScheduler(scheduler)

model.compile(optimizer = SGD(0.001, .9), loss= 'mse', callback = [callback])

r = model.fit(X,y, validation_split = 0.2, epochs = 200)

plt.plot(r.history['loss'], label = 'loss')


# In[12]:


## Let us get the values of W and b of y = WX + b

print(model.layers)
print(model.layers[0].get_weights())

a = model.layers[0].get_weights()[0][0,0]

print(a)

## So W is [[0.3550]] and b is [17.00]. Notice W is (1,1) matrix whereas b is (1,) vector. 
## By the way, W is the slope of the line

# So time for the transistor count to double is log(2)/a. See copy for full explanation. TBWD

print(" Time to double: ", np.log(2)/a)
# this is nearly equal to 2. every 2 years Moore's law doubles the number of transistors. Hence proved.

