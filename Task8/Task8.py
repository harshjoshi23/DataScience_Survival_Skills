#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras import layers, models


# ### Loading MNIST Data :

# In[2]:


(train_images, train_labels) , (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


# In[3]:


assert train_images.shape == (60000, 28, 28)
assert test_images.shape == (10000, 28, 28)
assert train_labels.shape == (60000,)
assert test_labels.shape == (10000,)


# ## EDA on Data : 

# In[4]:


print("Training set images shape:", train_images.shape)  
print("Training set labels shape:", train_labels.shape)
print("Test set images shape:", test_images.shape) 
print("Test set labels shape:", test_labels.shape)


# In[5]:


plt.figure(figsize=(10,10))
for i in range(25): 
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels)
plt.show()


# ### Checking the distribution of Labels : 

# In[6]:


unique, counts = np.unique(train_labels,return_counts=True)
print("Train labels distribution:", dict(zip(unique, counts)))

unique, counts = np.unique(test_labels, return_counts=True)
print("Test labels distribution: ", dict(zip(unique,counts)))


# In[ ]:





# ####  reshaping the data to include a channel dimension

# In[7]:


train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN input
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))


# In[ ]:





# ## Task 1 : 

# ### Selecting a Random Sample : 

# In[8]:


random_index = np.random.randint(0,len(train_images))
sample_image, sample_label = train_images[random_index], train_labels[random_index]


# In[9]:


sample_label


# ### Plotting the Sample : 

# In[10]:


plt.imshow(sample_image,cmap='gray')
plt.title(f'Label: {sample_label}')
plt.show()


# ### Examples of activation functions : 
# 
# Five commonly used activation functions in neural networks:
# 
# <b>ReLU (Rectified Linear Unit)</b><br>
# <b>Formula:</b> f(x) = max(0, x)<br>
# <b>Description:</b> ReLU is widely used due to its simplicity and efficiency. It outputs the input directly if it's positive, otherwise, it outputs zero.<br><br>
# 
# <b>Sigmoid</b><br>
# <b>Formula:</b> f(x) = 1 / (1 + e^(-x))<br>
# <b>Description:</b> The sigmoid function outputs values between 0 and 1, making it suitable for binary classification problems.<br><br>
# 
# <b>Tanh (Hyperbolic Tangent)</b><br>
# <b>Formula:</b> f(x) = (e^x - e^(-x)) / (e^x + e^(-x))<br>
# <b>Description:</b> Similar to sigmoid but outputs values between -1 and 1. Useful when the model needs to make decisions between two extremes.<br><br>
# 
# <b>Softmax</b><br>
# <b>Formula:</b> In a vector of real numbers x, the Softmax of the ith element is e^(x_i) / sum(e^(x_j))<br>
# <b>Description:</b> Often used in the output layer of a neural network for multi-class classification, returning probabilities of each class.<br><br>
# 
# <b>Leaky ReLU</b><br>
# <b>Formula:</b> f(x) = x if x > 0; f(x) = αx otherwise, where α is a small constant.<br>
# <b>Description:</b> A variation of ReLU that allows a small gradient when the unit is not active, helping to mitigate the dying ReLU problem.

# In[ ]:





# In[ ]:





# ## Task 2 : 

# #### Adding Convoluional Layres : 

# In[11]:


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# #### Flattening the output and adding Dense Layers: 

# In[12]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#### Compiling the Model : 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# ## Task 3 : 

# #### Optimizer Adam : 

# In[13]:


optimizer = 'adam'


# In[14]:


loss = 'sparse_categorical_crossentropy'


# In[15]:


metrics = ['accuracy']


# In[16]:


model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)


# ##### Second part

# In[17]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[18]:


model.summary()


# In[19]:


history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


# In[ ]:





# ## Task 4 : 

# #### Evaluating performance of our CNN model : 
# #### Plotting the training loss over Epochs : 

# In[20]:


plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[ ]:





# #### Evaluating Test accuracy : 

# In[21]:


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




