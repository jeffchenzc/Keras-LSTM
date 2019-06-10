#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re


# In[167]:


#This program predicts total power consumption of London in a day by examining past power consumptions and weather
f = open("part-00000", "r")

#retrive data from Spark output file
data = list()
output_shift = list()
for line in f:
    pur = re.sub(r'\(.*, \[|\]\)', '', line)
    split = pur.split(",")
    clean = np.array(split).astype(float)
    data.append(clean)
    output_shift.append(clean[0])

output_shift.insert(0, output_shift[0])
output_shift = output_shift[:-1]
output_shift = np.array([output_shift])
data = np.array(data)

total = len(data)
train_size = int(total * 0.6)
test_size = int(total * 0.3)
inp = train_size + test_size
ut_size = total - inp 
ut_size = ut_size - (ut_size % 4)

train, test, ut = data[:train_size], data[train_size:inp], data[inp:inp+ut_size]
x_test, y_test= test[:, 1:], test[:,0]
x_train, y_train = train[:, 1:], train[:, 0]
x_ut, y_ut = ut[:, 1:], ut[:, 0]

scale_size = 1000
y_test, y_train, y_ut, output_shift=    np.divide(y_test, scale_size), np.divide(y_train, scale_size), np.divide(y_ut, scale_size),    np.divide(output_shift, scale_size)


x_train = x_train.reshape((124,4,9))
y_train = y_train.reshape((124,4,1))[:,2,:]
x_test = x_test.reshape((62,4,9))
y_test = y_test.reshape((62,4,1))[:,2,:]

x_ut = x_ut.reshape((20, 4, 9))
y_ut = y_ut.reshape((20,4,1))[:,2,:]
    
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_ut = tf.keras.utils.normalize(x_ut, axis=1)

shift_train = output_shift[:,:train_size]
shift_train = shift_train.reshape(x_train.shape[0], 4, 1)
shift_test = output_shift[:,train_size:train_size+test_size]
shift_test = shift_test.reshape(x_test.shape[0], 4, 1)
shift_ut = output_shift[:,train_size+test_size:train_size+test_size+ut_size]
shift_ut = shift_ut.reshape(x_ut.shape[0], 4, 1)


x_train = np.dstack((shift_train, x_train))
x_test = np.dstack((shift_test, x_test))
x_ut = np.dstack((shift_ut, x_ut))

#building LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(80, return_sequences=False, input_shape=(4, 10)))
model.add(tf.keras.layers.Dense(20))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=90, validation_data=(x_test, y_test))


# In[168]:


#print loss of the last 10% unseen data and plot prediction(blue) against actual values(red)
#note: Unit of y axis is 1k
pred = model.predict(x_ut)
plt.plot(pred, color="blue")

against = y_ut
plt.plot(against, color="red")

vloss = model.evaluate(x_ut, y_ut)
print(vloss)


# In[ ]:




