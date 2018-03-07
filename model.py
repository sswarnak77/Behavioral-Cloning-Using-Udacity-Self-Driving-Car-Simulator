
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation
import cv2
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[2]:


import keras
print(keras.__version__)


# In[3]:


def network():
    model=Sequential()
    
    #Normalization Layer
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    
    # Convolutional Layer 1
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 3
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 5
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    #Flatten Layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 2
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 3
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(1))
    
    model.compile(loss="mse", optimizer="adam")
    
    return model


# In[4]:


def import_training_data(driving_csv_path):
    data=[]
    with open(driving_csv_path + '/driving_log.csv') as csvFile:
        reader=csv.reader(csvFile)
        next(reader)
        for line in reader:
            data.append(line)
    return data


# In[5]:


def getImageData(dataPath):
    directories = [x[0] for x in os.walk(dataPath)]
    print("directories",directories)
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataDirectories:
        print("directory",directory)
        lines = import_training_data(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)


# In[6]:


def combineImages(center, left, right, measurement, correction):
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)


# In[7]:


def dataPartition(samples,partition_ratio=0.2):
    train_data=[]
    validation_data=[]
    train_data,validation_data=train_test_split(samples,test_size=ratio)
    
    return train_data,validation_data


# In[8]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for image,angle in batch_samples:
                originalImage = cv2.imread(image)
                grey_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(grey_image)
                angles.append(angle)
                # Flipping
                images.append(cv2.flip(grey_image,1))
                angles.append(angle*-1.0)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[9]:


centerTotal, leftTotal, rightTotal, measurementTotal=getImageData('data')
imagePaths, measurements=combineImages(centerTotal,leftTotal,rightTotal,measurementTotal,0.2)
samples = list(zip(imagePaths, measurements))
train_data,validation_data=train_test_split(samples)


# In[10]:


centerTotal[:10]


# In[11]:


train_gen=generator(train_data)
validation_gen=generator(validation_data)


# In[12]:


validation_gen


# In[13]:


model=network()
model.fit_generator(train_gen,                                validation_steps=len(validation_data)/32,                                validation_data=validation_gen,                                steps_per_epoch=len(train_data)/32,                                epochs=3,verbose=1)


# In[14]:


import h5py


# In[15]:


model.save("final_retrained_mode.h5")

