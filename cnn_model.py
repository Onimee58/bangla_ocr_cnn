# -*- coding: utf-8 -*-
"""
Created on Mon May 10 04:24:41 2021

@author: Saif
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing import image                  
from keras.preprocessing.image import img_to_array, load_img
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True


def cnn_model(train_tensors):
    model = Sequential()
    
    # First Convolution Layer with Pooling
    model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='relu', input_shape=(train_tensors.shape[1:])))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Adding a second convolutional layer with Pooling
    model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size =2))
    model.add(Dropout(0.2))
    
    # Adding a third convolutional layer with Pooling
    model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size =2))
    model.add(Dropout(0.2))
    
    # Adding a fourth convolutional layer with Pooling
    model.add(Conv2D(filters=128, kernel_size=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size =2))
    model.add(Dropout(0.2))
    
    
    # Adding a fifth convolutional layer with Pooling
    model.add(Conv2D(filters=256, kernel_size=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size =2))
    model.add(Dropout(0.2))
    
    
    
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    
    # Full connection Dense Layers
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'softmax'))
    
    model.summary()
    return model

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(64, 64), grayscale=True)
    # convert PIL.Image.Image type to 3D tensor with shape (64, 64, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 64, 64, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


if __name__ == "__main__":
    cnn_model(path_to_tensor('trial/ka.jpg'))
