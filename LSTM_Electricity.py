#!/usr/bin/env python
# coding: utf-8

# In[152]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re


# In[168]:


#This program predicts total power consumption of London in a day by examining past power consumptions and weather
f = open("part-00000", "r")

#retrive data from Spark output file
data = list()
output_shift = list()
scale_size = 1000
limit = 10
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
ut_size = ut_size - (ut_size % 4)

pca = PCA(limit)
data = pca.fit_transform(data)

x_train, x_test, x_ut = data[:train_size], data[train_size:inp], data[inp:inp+ut_size]

print(x_train.shape, x_test.shape, x_ut.shape)

x_train = x_train.reshape((62,8,limit))
x_test = x_test.reshape((31,8,limit))
x_ut = x_ut.reshape((10, 8, limit))
    
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_ut = tf.keras.utils.normalize(x_ut, axis=1)

print(output_shift.shape)

y_train = output_shift[:,:train_size]
y_train = y_train.reshape(x_train.shape[0], 8, 1)[:,3,:]
y_test = output_shift[:,train_size:train_size+test_size]
y_test = y_test.reshape(x_test.shape[0], 8, 1)[:,3,:]
y_ut = output_shift[:,train_size+test_size:train_size+test_size+ut_size]
y_ut = y_ut.reshape(x_ut.shape[0], 8, 1)[:,3,:]

#building LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(1000, return_sequences=False, input_shape=(8, limit), dropout=0.2))
model.add(tf.keras.layers.Dense(100))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


# In[169]:


#print loss of the last 10% unseen data and plot prediction(blue) against actual values(red)
#note: Unit of y axis is 1k
plt.figure(0)

pred = model.predict(x_train)
plt.plot(pred, color="blue")
plt.plot(y_train, color="red")
vloss = model.evaluate(x_train, y_train)

plt.figure(1)

pred = model.predict(x_test)
plt.plot(pred, color="blue")
plt.plot(y_test, color="red")
vloss = model.evaluate(x_test, y_test)

plt.figure(2)

pred = model.predict(x_ut)
plt.plot(pred, color="blue")
plt.plot(y_ut, color="red")
vloss = model.evaluate(x_ut, y_ut)



# In[ ]:




