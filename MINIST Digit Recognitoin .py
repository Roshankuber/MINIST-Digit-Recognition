#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[4]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[8]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[9]:


x_train[0]


# In[19]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print(y_train[0])


# In[20]:


x_train=x_train/255
x_test=x_test/255


# In[21]:


x_train[0]


# In[23]:


model = Sequential()


# In[26]:


model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(10, activation="softmax"))


# In[27]:


model.summary()


# In[28]:


model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[29]:


history=model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.2)


# In[30]:


model.evaluate(x_test, y_test)


# In[31]:


model.predict(x_test)


# In[36]:


plt.imshow(x_test[0])


# In[39]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])


# In[ ]:




