#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re


# In[24]:


#This program predicts total power consumption of London in a day by examining past power consumptions and weather
f = open("part-00000", "r")

#retrive data from Spark output file
data = list()
output_shift = list()
scale_size = 1000
limit = 10
look_back = 4
for line in f:
    pur = re.sub(r'\(.*, \[|\]\)', '', line)
    split = pur.split(",")
    clean = np.array(split).astype(float)
    clean[0] /= scale_size
    data.append(clean)
    output_shift.append(clean[0])
output_shift.append(output_shift[-1])
output_shift = output_shift[1:]
output_shift = np.array([output_shift])
data = np.array(data)

total = len(data)
train_size = int(total * 0.6)
test_size = int(total * 0.3)
inp = train_size + test_size
ut_size = total - inp 
ut_size = ut_size - (ut_size % look_back)

pca = PCA(limit)
x_train, x_test, x_ut = data[:train_size], data[train_size:inp], data[inp:inp+ut_size]
x_train =pca.fit_transform(x_train)
means = pca.mean_
test, ut = x_test - means, x_ut - means
x_test, x_ut = np.dot(test, pca.components_.T), np.dot(ut, pca.components_.T)

print(x_train.shape, x_test.shape, x_ut.shape)

x_train = x_train.reshape((124,look_back,limit))
x_test = x_test.reshape((62,look_back,limit))
x_ut = x_ut.reshape((20, look_back, limit))
    
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_ut = tf.keras.utils.normalize(x_ut, axis=1)

print(output_shift.shape)

y_train = output_shift[:,:train_size]
y_train = y_train.reshape(x_train.shape[0], look_back, 1)#[:,3,:]
y_test = output_shift[:,train_size:train_size+test_size]
y_test = y_test.reshape(x_test.shape[0], look_back, 1)#[:,3,:]
y_ut = output_shift[:,train_size+test_size:train_size+test_size+ut_size]
y_ut = y_ut.reshape(x_ut.shape[0], look_back, 1)#[:,3,:]

#building LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(300, return_sequences=True, input_shape=(look_back, limit), dropout=0.1))
model.add(tf.keras.layers.LSTM(300, return_sequences=True, input_shape=(look_back, limit), dropout=0.2))
model.add(tf.keras.layers.Dense(100))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test))


# In[25]:


#print loss of the last 10% unseen data and plot prediction(blue) against actual values(red)
#note: Unit of y axis is 1k
plt.figure(0)

pred = model.predict(x_train)
plt.plot(pred[:,3,:], color="blue")
plt.plot(y_train[:,3,:], color="red")
vloss = model.evaluate(x_train, y_train)

plt.figure(1)

pred = model.predict(x_test)
plt.plot(pred[:,3,:], color="blue")
plt.plot(y_test[:,3,:], color="red")
vloss = model.evaluate(x_test, y_test)

plt.figure(2)

pred = model.predict(x_ut)
plt.plot(pred[:,3,:], color="blue")
plt.plot(y_ut[:,3,:], color="red")
vloss = model.evaluate(x_ut, y_ut)



# In[ ]:




