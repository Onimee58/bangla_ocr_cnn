# -*- coding: utf-8 -*-
"""
Created on Mon May 10 04:24:07 2021

@author: Saif
"""

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
from keras.preprocessing import image                  
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from PIL import ImageFile                           
ImageFile.LOAD_TRUNCATED_IMAGES = True 

model = load_model('saved_models/weights.best.with_augmentation_new.hdf5')

def trial_prediction(img_path):
    img = load_img(img_path, target_size=(64, 64), grayscale=True)
    x = img_to_array(img)
    tensor = x #np.expand_dims(x, axis=0)
    test_img = np.expand_dims(tensor, axis=0)
    prediction_idx = np.argmax(model.predict(test_img))
    alphbt = ['অ','আ','ই','ঈ','উ','ঊ',
              'ঋ','এ','ঐ','ও','ঔ',
              'ক','খ','গ','ঘ','ঙ',
              'চ','ছ','জ','ঝ','ঞ',
              'ট','ঠ','ড','ঢ','ণ',
              'ত','থ','দ','ধ','ন',
              'প','ফ','ব','ভ','ম',
              'য','র','ল',
              'শ','ষ','স','হ',
              'ড়','ঢ়','য়',
              'ৎ','ং','ঃ','ঁ'
              'ঁ']
    output = alphbt[prediction_idx]
    return output

#%% run this only

print("Enter the file path for image.\nPress Ctrl+C to exit.")

while(True):
    loc = input('Image path: ')
    out = trial_prediction(loc)
    print(out)
