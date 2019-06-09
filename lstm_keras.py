#!/usr/bin/env python
# coding: utf-8

# In[313]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re


# In[392]:



f = open("part-00000", "r")

#retrive data from Spark output file
data = list()
example = list()
for line in f:
    example.append(line)
    pur = re.sub(r'\(.*, \[|\]\)', '', line)
    split = pur.split(",")
    clean = np.array(split).astype(float)
    data.append(clean)
data = np.array(data)
total = len(data)
train_size = int(total * 0.6)
test_size = int(total * 0.3)
inp = train_size + test_size
ut_size = total - inp 
ut_size = ut_size - (ut_size % 4)
print(total, train_size, test_size, ut_size)

train, test, ut = data[:train_size], data[train_size:inp], data[inp:inp+ut_size]
x_test, y_test= test[:, 1:], test[:,0]
x_train, y_train = train[:, 1:], train[:, 0]
x_ut, y_ut = ut[:, 1:], ut[:, 0]

y_test, y_train, y_ut = np.divide(y_test, 1000), np.divide(y_train, 1000), np.divide(y_ut, 1000)

x_train = x_train.reshape((124,4,9))
y_train = y_train.reshape((124,4,1))[:,2,:]
#y_train = convertY(y_train)
x_test = x_test.reshape((62,4,9))
y_test = y_test.reshape((62,4,1))[:,2,:]

x_ut = x_ut.reshape((20, 4, 9))
y_ut = y_ut.reshape((20,4,1))[:,2,:]

# y_test, y_train = tf.keras.utils.normalize(y_test, axis=1),\
#     tf.keras.utils.normalize(y_train, axis=1)
    
x_test = tf.keras.utils.normalize(x_test, axis=0)
x_train = tf.keras.utils.normalize(x_train, axis=0)
x_ut = tf.keras.utils.normalize(x_ut, axis=0)

#building LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(1000, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.2))
model.add(tf.keras.layers.Dense(100))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=90, validation_data=(x_test, y_test))


# In[395]:


# xreshape = x_test[14].reshape((1,3,9))
# yreshape = y_test[14].reshape((1,1))
pred = model.predict(x_test)
# pred = pred[:,2,:]
# pred = pred.reshape((pred.shape[0], 1))
plt.plot(pred, color="blue")

against = y_test
# against = against[:,2,:]
# against = against.reshape((against.shape[0], 1))
plt.plot(against, color="red")
#plt.plot(y_test)
#print(loss, acc)


# In[ ]:




